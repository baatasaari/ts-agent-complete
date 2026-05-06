"""
ts_agent.zones.session_resume
==============================
Session Resume Service — handles mid-conversation disconnection (Section 11.1).

Responsibilities
----------------
- Detect whether a resumable session exists for a returning consumer.
- Reconstruct TraitGraph from Neo4j using Pattern 1 traversal.
- Restore the last active SegmentHypothesis.
- Return the next unanswered question (honoring INV-09).
- Emit all SESSION_* EAMGP signals.

NOT responsible for
-------------------
- Authentication (handled by Apigee JWT layer upstream).
- Re-running Zone 1 bank-data population (caller decides if needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from ts_agent.domain.models import (
    NodeState,
    SegmentHypothesis,
    SessionDisposition,
    TraitGraph,
    TraitNode,
)
from ts_agent.observability import signals as eamgp

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

class ResumeAction(str, Enum):
    RESUMED     = "RESUMED"
    NEW_SESSION = "NEW_SESSION"
    FAILED      = "FAILED"


@dataclass
class ResumeResult:
    action: ResumeAction
    session_id: str = ""
    graph: TraitGraph | None = None
    hypothesis: SegmentHypothesis | None = None
    next_question_char_id: str | None = None
    already_known_count: int = 0
    reason: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Repository protocol (injectable)
# ──────────────────────────────────────────────────────────────────────────────

class SessionRepository(Protocol):
    """Persistence interface used by the resume service."""

    async def find_resumable_session(
        self,
        party_ref: str,
        resume_window: timedelta,
    ) -> dict[str, Any] | None:
        """
        Return the most recent ACTIVE / PAUSED / DISCONNECTED session
        within the resume window, or ``None``.
        """
        ...

    async def reconstruct_graph(self, session_id: str) -> TraitGraph | None:
        """Pattern 1 — reconstruct full TraitGraph from Neo4j."""
        ...

    async def get_active_hypothesis(
        self, session_id: str
    ) -> SegmentHypothesis | None:
        """Return the current ACTIVE :PredictedSegment for a session."""
        ...

    async def write_reconnect_event(self, session_id: str) -> None:
        """Append a RECONNECTED marker to the session node in Neo4j."""
        ...

    async def expire_session(self, session_id: str) -> None:
        """Mark an expired session so it is excluded from future lookups."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# SessionResumeService
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_RESUME_WINDOW = timedelta(hours=24)


class SessionResumeService:
    """
    Handles consumer reconnection at any point in the conversation.

    Usage::

        result = await service.resume(party_ref="PARTY-123")
        if result.action == ResumeAction.RESUMED:
            # Continue from result.next_question_char_id
        else:
            # Start a new session
    """

    def __init__(
        self,
        repository: SessionRepository,
        resume_window: timedelta = _DEFAULT_RESUME_WINDOW,
    ) -> None:
        self._repo = repository
        self._resume_window = resume_window

    async def resume(self, party_ref: str) -> ResumeResult:
        """
        Attempt to resume the most recent session for ``party_ref``.

        Returns a ``ResumeResult`` describing the action taken and,
        if resumed, the reconstructed state.
        """
        try:
            return await self._do_resume(party_ref)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Session resume raised unexpected error: %s", exc)
            eamgp.emit(
                "SESSION_RESUME_FAILED",
                eamgp.ERROR,
                "Session",
                party_ref=party_ref,
                error_type=type(exc).__name__,
                fallback="NEW_SESSION",
            )
            return ResumeResult(
                action=ResumeAction.FAILED,
                reason=str(exc),
            )

    async def _do_resume(self, party_ref: str) -> ResumeResult:
        # 1. Find a session in the resume window
        session_data = await self._repo.find_resumable_session(
            party_ref, self._resume_window
        )

        if session_data is None:
            return ResumeResult(
                action=ResumeAction.NEW_SESSION,
                reason="NO_RESUMABLE_SESSION",
            )

        session_id = session_data["session_id"]
        created_ts = session_data.get("created_ts")

        # 2. TTL check (belt-and-braces — the query should filter, but verify)
        if created_ts and self._is_expired(created_ts):
            await self._repo.expire_session(session_id)
            eamgp.emit(
                "SESSION_EXPIRED",
                eamgp.WARN,
                "Session",
                session_id=session_id,
                party_ref=party_ref,
            )
            return ResumeResult(
                action=ResumeAction.NEW_SESSION,
                reason="SESSION_EXPIRED",
            )

        # 3. Reconstruct graph
        graph = await self._repo.reconstruct_graph(session_id)
        if graph is None:
            eamgp.emit(
                "SESSION_RESUME_FAILED",
                eamgp.ERROR,
                "Session",
                session_id=session_id,
                party_ref=party_ref,
                error_type="GraphReconstructionFailed",
                fallback="NEW_SESSION",
            )
            return ResumeResult(
                action=ResumeAction.NEW_SESSION,
                reason="GRAPH_RECONSTRUCTION_FAILED",
            )

        # 4. Restore hypothesis chain
        hypothesis = await self._repo.get_active_hypothesis(session_id)

        # 5. Determine next unanswered question (INV-09: never re-ask KNOWN)
        next_char_id = self._next_question(graph)

        # 6. Write reconnect marker
        await self._repo.write_reconnect_event(session_id)

        already_known = len(graph.known_nodes())

        eamgp.emit(
            "SESSION_RESUMED",
            eamgp.INFO,
            "Session",
            session_id=session_id,
            party_ref=party_ref,
            traits_already_known=already_known,
            next_question_char_id=next_char_id,
        )

        return ResumeResult(
            action=ResumeAction.RESUMED,
            session_id=session_id,
            graph=graph,
            hypothesis=hypothesis,
            next_question_char_id=next_char_id,
            already_known_count=already_known,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_expired(self, created_ts: datetime | str) -> bool:
        if isinstance(created_ts, str):
            created_ts = datetime.fromisoformat(created_ts)
        if created_ts.tzinfo is None:
            created_ts = created_ts.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - created_ts > self._resume_window

    @staticmethod
    def _next_question(graph: TraitGraph) -> str | None:
        """
        Return the char_id of the highest-priority MISSING node.
        INV-09: only MISSING nodes are candidates — KNOWN nodes are never
        re-asked regardless of reconnection.
        """
        missing = graph.missing_nodes()
        if not missing:
            return None
        return min(missing, key=lambda n: n.fill_priority).char_id
