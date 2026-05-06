"""
tests/unit/zones/test_session_resume.py
=======================================
Unit tests for ts_agent.zones.session_resume

All repository interactions are stubbed — no Neo4j required.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ts_agent.zones.session_resume import (
    ResumeAction,
    SessionResumeService,
)
from tests.fixtures.factories import (
    make_complete_graph,
    make_hypothesis,
    make_incomplete_graph,
)


# ──────────────────────────────────────────────────────────────────────────────
# Repository stubs
# ──────────────────────────────────────────────────────────────────────────────

class StubRepository:
    """In-memory stub for SessionRepository."""

    def __init__(
        self,
        session_data: dict | None = None,
        graph=None,
        hypothesis=None,
        raise_on_reconstruct: Exception | None = None,
    ):
        self._session_data       = session_data
        self._graph              = graph
        self._hypothesis         = hypothesis
        self._raise_reconstruct  = raise_on_reconstruct
        self.reconnect_events: list = []
        self.expired: list          = []

    async def find_resumable_session(self, party_ref, resume_window):
        return self._session_data

    async def reconstruct_graph(self, session_id):
        if self._raise_reconstruct:
            raise self._raise_reconstruct
        return self._graph

    async def get_active_hypothesis(self, session_id):
        return self._hypothesis

    async def write_reconnect_event(self, session_id):
        self.reconnect_events.append(session_id)

    async def expire_session(self, session_id):
        self.expired.append(session_id)


def _active_session(session_id: str = "sess-001", age_hours: float = 0.0) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return {"session_id": session_id, "created_ts": ts}


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionResumeService:

    @pytest.fixture()
    def service(self) -> SessionResumeService:
        repo = StubRepository()
        return SessionResumeService(repo)

    # ── No session found ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_returns_new_session_when_no_resumable_session(self):
        repo    = StubRepository(session_data=None)
        service = SessionResumeService(repo)
        result  = await service.resume("PARTY-001")
        assert result.action == ResumeAction.NEW_SESSION
        assert result.reason == "NO_RESUMABLE_SESSION"

    # ── TTL expiry ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_expired_session_returns_new_session(self):
        repo = StubRepository(
            session_data=_active_session(age_hours=25),  # > 24h
        )
        service = SessionResumeService(repo, resume_window=timedelta(hours=24))
        result  = await service.resume("PARTY-001")
        assert result.action == ResumeAction.NEW_SESSION
        assert result.reason == "SESSION_EXPIRED"

    @pytest.mark.asyncio
    async def test_expired_session_marks_as_expired_in_repo(self):
        repo = StubRepository(
            session_data=_active_session(session_id="sess-old", age_hours=30),
        )
        service = SessionResumeService(repo, resume_window=timedelta(hours=24))
        await service.resume("PARTY-001")
        assert "sess-old" in repo.expired

    @pytest.mark.asyncio
    async def test_session_within_window_is_not_expired(self):
        repo = StubRepository(
            session_data=_active_session(age_hours=1),
            graph=make_complete_graph(),
        )
        service = SessionResumeService(repo, resume_window=timedelta(hours=24))
        result  = await service.resume("PARTY-001")
        assert result.action == ResumeAction.RESUMED

    # ── Successful resume ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_resumed_result_contains_session_id(self):
        repo = StubRepository(
            session_data=_active_session(session_id="sess-42"),
            graph=make_complete_graph(session_id="sess-42"),
        )
        service = SessionResumeService(repo)
        result  = await service.resume("PARTY-001")
        assert result.session_id == "sess-42"

    @pytest.mark.asyncio
    async def test_resumed_result_contains_graph(self):
        graph = make_complete_graph()
        repo  = StubRepository(session_data=_active_session(), graph=graph)
        result = await SessionResumeService(repo).resume("PARTY-001")
        assert result.graph is graph

    @pytest.mark.asyncio
    async def test_resumed_result_contains_hypothesis(self):
        hyp  = make_hypothesis()
        repo = StubRepository(
            session_data=_active_session(),
            graph=make_complete_graph(),
            hypothesis=hyp,
        )
        result = await SessionResumeService(repo).resume("PARTY-001")
        assert result.hypothesis is hyp

    @pytest.mark.asyncio
    async def test_already_known_count_reported(self):
        graph = make_complete_graph()
        known_count = len(graph.known_nodes())
        repo  = StubRepository(session_data=_active_session(), graph=graph)
        result = await SessionResumeService(repo).resume("PARTY-001")
        assert result.already_known_count == known_count

    # ── INV-09: next question never re-asks known traits ─────────────────────

    @pytest.mark.asyncio
    async def test_next_question_is_a_missing_node(self):
        graph = make_incomplete_graph()
        repo  = StubRepository(session_data=_active_session(), graph=graph)
        result = await SessionResumeService(repo).resume("PARTY-001")
        # next_question_char_id must be a MISSING node, not a KNOWN one
        if result.next_question_char_id is not None:
            missing_char_ids = {n.char_id for n in graph.missing_nodes()}
            assert result.next_question_char_id in missing_char_ids

    @pytest.mark.asyncio
    async def test_next_question_is_none_when_graph_is_complete(self):
        graph = make_complete_graph()
        repo  = StubRepository(session_data=_active_session(), graph=graph)
        result = await SessionResumeService(repo).resume("PARTY-001")
        assert result.next_question_char_id is None

    @pytest.mark.asyncio
    async def test_reconnect_event_written_on_resume(self):
        repo = StubRepository(
            session_data=_active_session(session_id="sess-rec"),
            graph=make_complete_graph(),
        )
        service = SessionResumeService(repo)
        await service.resume("PARTY-001")
        assert "sess-rec" in repo.reconnect_events

    # ── Graph reconstruction failure ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_returns_new_session_when_graph_reconstruction_returns_none(self):
        repo = StubRepository(
            session_data=_active_session(),
            graph=None,    # reconstruction returns None
        )
        result = await SessionResumeService(repo).resume("PARTY-001")
        assert result.action == ResumeAction.NEW_SESSION
        assert result.reason == "GRAPH_RECONSTRUCTION_FAILED"

    # ── Unexpected errors ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_failed_action(self):
        repo = StubRepository(raise_on_reconstruct=RuntimeError("DB exploded"))
        repo._session_data = _active_session()
        result = await SessionResumeService(repo).resume("PARTY-001")
        assert result.action == ResumeAction.FAILED

    # ── EAMGP signals ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_session_resumed_signal_emitted(self, mocker):
        mock_emit = mocker.patch("ts_agent.zones.session_resume.eamgp.emit")
        repo = StubRepository(
            session_data=_active_session(), graph=make_complete_graph()
        )
        await SessionResumeService(repo).resume("PARTY-001")
        signals = [c.args[0] for c in mock_emit.call_args_list]
        assert "SESSION_RESUMED" in signals

    @pytest.mark.asyncio
    async def test_session_expired_signal_emitted_on_ttl(self, mocker):
        mock_emit = mocker.patch("ts_agent.zones.session_resume.eamgp.emit")
        repo = StubRepository(session_data=_active_session(age_hours=25))
        service = SessionResumeService(repo, resume_window=timedelta(hours=24))
        await service.resume("PARTY-001")
        signals = [c.args[0] for c in mock_emit.call_args_list]
        assert "SESSION_EXPIRED" in signals

    @pytest.mark.asyncio
    async def test_session_resume_failed_signal_emitted_on_exception(self, mocker):
        mock_emit = mocker.patch("ts_agent.zones.session_resume.eamgp.emit")
        repo = StubRepository(raise_on_reconstruct=RuntimeError("boom"))
        repo._session_data = _active_session()
        await SessionResumeService(repo).resume("PARTY-001")
        signals = [c.args[0] for c in mock_emit.call_args_list]
        assert "SESSION_RESUME_FAILED" in signals
