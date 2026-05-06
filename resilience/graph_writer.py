"""
ts_agent.resilience.graph_writer
=================================
Resilient Neo4j write layer with circuit breaker, exponential back-off,
and dead-letter queue (Pub/Sub) for soft writes.

Write path priority (Section 11.3.1)
--------------------------------------
HARD  — Zone 1 TraitGraph write, MATCHED_SEGMENT edge, LINKED_TO_SUGG edge.
        Raises ``Neo4jWriteError`` after exhausting retries.
        Circuit breaker opens after ``circuit_threshold`` consecutive failures.

SOFT  — SegmentHypothesis writes, analytics edges.
        Failures are caught, logged, and pushed to Pub/Sub DLQ.
        The caller is never blocked.

Design decisions
----------------
- The circuit breaker is *per-instance*, i.e. one writer per GKE pod.
  In production a shared Redis counter would be preferable, but that adds
  a dependency; for now the pod-local circuit is correct for the p99 latency
  budget and accepted in the design document.
- All Cypher is parameterised (injection-safe).
- The writer does not import ``neo4j`` at module load time so the module
  can be imported in test environments without a Neo4j driver.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from ts_agent.domain.models import SegmentHypothesis, TraitGraph
from ts_agent.config.settings import settings
from ts_agent.observability import signals as eamgp

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class Neo4jWriteError(RuntimeError):
    """Raised when a HARD write fails after all retries."""


class Neo4jCircuitOpenError(Neo4jWriteError):
    """Raised when the circuit breaker is OPEN and a HARD write is attempted."""


# ──────────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ──────────────────────────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreaker:
    """
    Simple three-state circuit breaker.

    Parameters
    ----------
    failure_threshold   : consecutive failures before opening.
    recovery_timeout_s  : seconds in OPEN state before moving to HALF_OPEN.
    """
    failure_threshold:  int   = 5
    recovery_timeout_s: float = 60.0

    _state:            CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count:    int          = field(default=0,                  init=False)
    _last_failure_ts:  float        = field(default=0.0,                init=False)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_ts = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def is_open(self) -> bool:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_ts
            if elapsed >= self.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                return False
            return True
        return False

    @property
    def state(self) -> CircuitState:
        return self._state


# ──────────────────────────────────────────────────────────────────────────────
# DLQ publisher protocol (injectable)
# ──────────────────────────────────────────────────────────────────────────────

class DLQPublisher(Protocol):
    """Interface for pushing failed payloads to the dead-letter queue."""

    async def publish(self, message: dict[str, Any]) -> None:
        ...


class NoopDLQPublisher:
    """Stub used in tests and local dev where Pub/Sub is unavailable."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish(self, message: dict[str, Any]) -> None:
        self.published.append(message)
        logger.debug("DLQ (noop): %s", message.get("operation"))


# ──────────────────────────────────────────────────────────────────────────────
# Neo4j driver protocol (injectable)
# ──────────────────────────────────────────────────────────────────────────────

class Neo4jDriver(Protocol):
    """Minimal interface the writer needs from the Neo4j driver."""

    async def execute_write(
        self,
        cypher: str,
        parameters: dict[str, Any],
    ) -> Any:
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Cypher statements
# ──────────────────────────────────────────────────────────────────────────────

_UPSERT_SESSION_CYPHER = """
MERGE (s:Session {session_id: $session_id})
ON CREATE SET
    s.party_ref    = $party_ref,
    s.intent_id    = $intent_id,
    s.situation_id = $situation_id,
    s.created_ts   = datetime(),
    s.ttl_ts       = datetime() + duration({days: 90}),
    s.graph_version= $graph_version,
    s.disposition  = 'ACTIVE'
RETURN s.session_id AS sid
"""

_UPSERT_NODES_CYPHER = """
UNWIND $rows AS row
MERGE (n:TraitNode {node_id: row.nodeId})
SET n += {
    charId:          row.charId,
    branch:          row.branch,
    label:           row.label,
    op:              row.op,
    targetValue:     row.targetValue,
    state:           row.state,
    value:           row.value,
    populatedSource: row.populatedSource,
    fcaRef:          row.fcaRef
}
WITH n, row
MATCH (s:Session {session_id: $session_id})
MERGE (s)-[:HAS_TRAIT]->(n)
"""

_UPSERT_TYPED_EDGES_CYPHER = """
UNWIND $rows AS row
MATCH (a {node_id: row.fromId}), (b {node_id: row.toId})
MERGE (a)-[r:%(rel_type)s]->(b)
ON CREATE SET r = row.props
"""

_PERSIST_HYPOTHESIS_CYPHER = """
MATCH (s:Session {session_id: $session_id})
OPTIONAL MATCH (s)-[:HAS_PREDICTION]->(prev:PredictedSegment {disposition: 'ACTIVE'})
FOREACH (p IN CASE WHEN prev IS NOT NULL THEN [prev] ELSE [] END |
    SET p.disposition = 'SUPERSEDED'
)
CREATE (h:PredictedSegment {
    hypothesis_id:     $hypothesis_id,
    session_id:        $session_id,
    turn:              $turn,
    top_segment_id:    $top_segment_id,
    top_confidence:    $top_confidence,
    shap_features:     $shap_json,
    model_version:     $model_version,
    model_algorithm:   $model_algorithm,
    known_trait_count: $known_trait_count,
    disposition:       $disposition,
    created_ts:        datetime()
})
MERGE (s)-[:HAS_PREDICTION]->(h)
WITH h, prev
FOREACH (p IN CASE WHEN prev IS NOT NULL THEN [prev] ELSE [] END |
    MERGE (h)-[:SUPERSEDES]->(p)
)
WITH h
OPTIONAL MATCH (seg:Segment {segment_id: $top_segment_id})
FOREACH (seg IN CASE WHEN seg IS NOT NULL THEN [seg] ELSE [] END |
    MERGE (h)-[:PREDICTED_SEGMENT]->(seg)
)
RETURN h.hypothesis_id AS hid
"""


# ──────────────────────────────────────────────────────────────────────────────
# ResilientNeo4jWriter
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_RETRY_DELAYS: tuple[float, ...] = (0.1, 0.5, 2.0)


class ResilientNeo4jWriter:
    """
    HARD and SOFT write paths with retry, circuit breaker, and DLQ.

    Inject this into Zone 1 (TraitGraphBuilder) and Zone 1.5
    (IterativeSegmentPredictor) via the ``GraphWriter`` protocol.
    """

    def __init__(
        self,
        driver: Neo4jDriver,
        dlq_publisher: DLQPublisher,
        retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS,
        circuit_threshold: int = 5,
        recovery_timeout_s: float = 60.0,
    ) -> None:
        self._driver  = driver
        self._dlq     = dlq_publisher
        self._delays  = retry_delays
        self._circuit = CircuitBreaker(
            failure_threshold=circuit_threshold,
            recovery_timeout_s=recovery_timeout_s,
        )

    # ── HARD write ────────────────────────────────────────────────────────────

    async def write_hard(self, graph: TraitGraph) -> None:
        """
        Persist the full session TraitGraph to Neo4j.

        Raises ``Neo4jCircuitOpenError`` if the circuit is OPEN.
        Raises ``Neo4jWriteError`` after exhausting retries.
        Emits ``NEO4J_WRITE_HARD_FAIL`` on failure.
        """
        if self._circuit.is_open():
            eamgp.emit(
                "NEO4J_WRITE_HARD_FAIL",
                eamgp.ERROR,
                "Zone1",
                session_id=graph.session_id,
                operation="write_hard",
                attempt_count=0,
                error_type="CircuitOpen",
                circuit_state=self._circuit.state.value,
            )
            raise Neo4jCircuitOpenError(
                f"ERR-GW-006: Neo4j circuit {self._circuit.state.value} "
                f"— Zone 1 write denied"
            )

        last_exc: Exception | None = None
        for attempt, delay in enumerate(self._delays, start=1):
            try:
                await self._do_write_graph(graph)
                self._circuit.record_success()
                if self._circuit.state == CircuitState.CLOSED:
                    pass  # normal — no extra signal
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._circuit.record_failure()
                logger.warning(
                    "Neo4j hard write attempt %d/%d failed: %s",
                    attempt,
                    len(self._delays),
                    exc,
                )
                if attempt < len(self._delays):
                    await asyncio.sleep(delay)

        eamgp.emit(
            "NEO4J_WRITE_HARD_FAIL",
            eamgp.ERROR,
            "Zone1",
            session_id=graph.session_id,
            operation="write_hard",
            attempt_count=len(self._delays),
            error_type=type(last_exc).__name__,
            circuit_state=self._circuit.state.value,
        )
        if self._circuit.is_open():
            eamgp.emit("NEO4J_CIRCUIT_OPENED", eamgp.ERROR, "Zone1",
                       failure_count=self._circuit._failure_count,
                       last_error=str(last_exc))
        raise Neo4jWriteError(
            f"ERR-GW-001: Neo4j hard write failed after {len(self._delays)} "
            f"attempts — {last_exc}"
        ) from last_exc

    # ── SOFT write (hypothesis) ───────────────────────────────────────────────

    async def write_hypothesis(self, hyp: SegmentHypothesis) -> None:
        """
        Persist a SegmentHypothesis to Neo4j.

        Failure never blocks inference; payload is routed to DLQ.
        """
        params = hyp.to_neo4j_params()
        try:
            await self._driver.execute_write(_PERSIST_HYPOTHESIS_CYPHER, params)
            eamgp.emit(
                "SEG_HYPOTHESIS_WRITTEN",
                eamgp.INFO,
                "Zone1.5",
                session_id=hyp.session_id,
                hypothesis_id=hyp.hypothesis_id,
            )
        except Exception as exc:  # noqa: BLE001
            eamgp.emit(
                "SEG_HYPOTHESIS_WRITE_FAIL",
                eamgp.ERROR,
                "Zone1.5",
                session_id=hyp.session_id,
                hypothesis_id=hyp.hypothesis_id,
                error_type=type(exc).__name__,
            )
            await self._route_to_dlq(
                operation="write_hypothesis",
                session_id=hyp.session_id,
                payload=params,
                error=str(exc),
            )
            # Do NOT re-raise — SOFT write failure is non-blocking

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _do_write_graph(self, graph: TraitGraph) -> None:
        """Execute the three-step graph write in sequence."""
        # 1. Session node
        await self._driver.execute_write(
            _UPSERT_SESSION_CYPHER,
            {
                "session_id":    graph.session_id,
                "party_ref":     graph.party_ref,
                "intent_id":     graph.intent_id,
                "situation_id":  graph.situation_id,
                "graph_version": graph.graph_version,
            },
        )
        # 2. TraitNode bulk upsert
        node_rows = [
            {
                "nodeId":          n.node_id,
                "charId":          n.char_id,
                "branch":          n.branch.value,
                "label":           n.label,
                "op":              n.op,
                "targetValue":     str(n.target_value),
                "state":           n.state.value,
                "value":           str(n.value) if n.value is not None else None,
                "populatedSource": n.populated_source or "",
                "fcaRef":          n.fca_ref or "",
            }
            for n in graph.nodes.values()
        ]
        await self._driver.execute_write(
            _UPSERT_NODES_CYPHER,
            {"rows": node_rows, "session_id": graph.session_id},
        )
        # 3. Typed edges (exclude HAS_TRAIT — already created in step 2)
        for edge_type, edges in self._group_edges(graph):
            if edge_type == "HAS_TRAIT":
                continue
            rows = [
                {
                    "fromId": e.from_id,
                    "toId":   e.to_id,
                    "props":  e.properties,
                }
                for e in edges
            ]
            cypher = _UPSERT_TYPED_EDGES_CYPHER % {"rel_type": edge_type}
            await self._driver.execute_write(cypher, {"rows": rows})

    @staticmethod
    def _group_edges(
        graph: TraitGraph,
    ) -> list[tuple[str, list]]:
        groups: dict[str, list] = {}
        for edge in graph.edges:
            groups.setdefault(edge.edge_type.value, []).append(edge)
        return list(groups.items())

    async def _route_to_dlq(
        self,
        operation: str,
        session_id: str,
        payload: dict[str, Any],
        error: str,
    ) -> None:
        import time as _t  # noqa: PLC0415
        await self._dlq.publish({
            "operation":   operation,
            "session_id":  session_id,
            "payload":     payload,
            "error":       error,
            "ts":          str(_t.time()),
            "retry_count": 0,
        })
        eamgp.emit(
            "NEO4J_SOFT_WRITE_FAILED",
            eamgp.WARN,
            "Infra",
            session_id=session_id,
            operation=operation,
            dlq_published=True,
        )

    # ── Health / state ────────────────────────────────────────────────────────

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit.state
