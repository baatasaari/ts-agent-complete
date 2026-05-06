"""
tests/unit/resilience/test_graph_writer.py
==========================================
Unit tests for ts_agent.resilience.graph_writer

Tests cover circuit breaker state transitions, retry behaviour,
DLQ routing, and EAMGP signal emission.  No real Neo4j driver used.
"""

import asyncio
import pytest

from ts_agent.domain.models import HypothesisDisposition, ModelAlgorithm
from ts_agent.resilience.graph_writer import (
    CircuitBreaker,
    CircuitState,
    Neo4jCircuitOpenError,
    Neo4jWriteError,
    NoopDLQPublisher,
    ResilientNeo4jWriter,
)
from tests.fixtures.factories import make_complete_graph, make_hypothesis


# ──────────────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────────────

class OkDriver:
    """Driver that always succeeds."""
    def __init__(self):
        self.calls: list = []

    async def execute_write(self, cypher, parameters):
        self.calls.append((cypher, parameters))
        return {"ok": True}


class FailingDriver:
    """Driver that always raises."""
    def __init__(self, error: Exception | None = None):
        self.calls = 0
        self._error = error or RuntimeError("Neo4j unavailable")

    async def execute_write(self, cypher, parameters):
        self.calls += 1
        raise self._error


class FlakyDriver:
    """Driver that fails the first N calls then succeeds."""
    def __init__(self, fail_count: int = 1):
        self.attempts  = 0
        self.fail_count = fail_count

    async def execute_write(self, cypher, parameters):
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise RuntimeError("Transient error")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# CircuitBreaker
# ──────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open() is False

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open() is True

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Only 2 failures since last success → still CLOSED
        assert cb.state == CircuitState.CLOSED

    def test_moves_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # With timeout=0, is_open() should transition to HALF_OPEN
        result = cb.is_open()
        assert result is False
        assert cb.state == CircuitState.HALF_OPEN


# ──────────────────────────────────────────────────────────────────────────────
# ResilientNeo4jWriter — HARD writes
# ──────────────────────────────────────────────────────────────────────────────

class TestResilientNeo4jWriterHard:

    @pytest.fixture()
    def dlq(self) -> NoopDLQPublisher:
        return NoopDLQPublisher()

    @pytest.mark.asyncio
    async def test_successful_hard_write_calls_driver(self, dlq):
        driver = OkDriver()
        writer = ResilientNeo4jWriter(driver, dlq)
        graph  = make_complete_graph()
        await writer.write_hard(graph)
        assert len(driver.calls) > 0

    @pytest.mark.asyncio
    async def test_circuit_remains_closed_after_success(self, dlq):
        driver = OkDriver()
        writer = ResilientNeo4jWriter(driver, dlq)
        await writer.write_hard(make_complete_graph())
        assert writer.circuit_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_hard_write_retries_on_transient_error(self, dlq):
        driver = FlakyDriver(fail_count=1)
        writer = ResilientNeo4jWriter(
            driver, dlq, retry_delays=(0.0, 0.0, 0.0)
        )
        # Should succeed on second attempt without raising
        await writer.write_hard(make_complete_graph())
        assert driver.attempts >= 2

    @pytest.mark.asyncio
    async def test_hard_write_raises_after_all_retries(self, dlq):
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(
            driver, dlq, retry_delays=(0.0, 0.0)
        )
        with pytest.raises(Neo4jWriteError, match="ERR-GW-001"):
            await writer.write_hard(make_complete_graph())

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self, dlq):
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(
            driver, dlq,
            retry_delays=(0.0,),       # 1 retry only
            circuit_threshold=2,
        )
        # Two hard-write failures → circuit opens
        for _ in range(2):
            try:
                await writer.write_hard(make_complete_graph())
            except Neo4jWriteError:
                pass
        assert writer.circuit_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_raises_immediately(self, dlq):
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(
            driver, dlq,
            retry_delays=(0.0,),
            circuit_threshold=1,
        )
        # First call opens circuit
        try:
            await writer.write_hard(make_complete_graph())
        except Neo4jWriteError:
            pass

        assert writer.circuit_state == CircuitState.OPEN
        # Second call should raise immediately without hitting the driver again
        call_count_before = driver.calls
        with pytest.raises(Neo4jCircuitOpenError, match="ERR-GW-006"):
            await writer.write_hard(make_complete_graph())
        # Driver was not called again
        assert driver.calls == call_count_before

    @pytest.mark.asyncio
    async def test_hard_write_emits_fail_signal_on_exhaustion(self, dlq, mocker):
        mock_emit = mocker.patch("ts_agent.resilience.graph_writer.eamgp.emit")
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(driver, dlq, retry_delays=(0.0,))
        with pytest.raises(Neo4jWriteError):
            await writer.write_hard(make_complete_graph())
        signals = [call.args[0] for call in mock_emit.call_args_list]
        assert "NEO4J_WRITE_HARD_FAIL" in signals


# ──────────────────────────────────────────────────────────────────────────────
# ResilientNeo4jWriter — SOFT writes (hypothesis)
# ──────────────────────────────────────────────────────────────────────────────

class TestResilientNeo4jWriterSoft:

    @pytest.fixture()
    def dlq(self) -> NoopDLQPublisher:
        return NoopDLQPublisher()

    @pytest.mark.asyncio
    async def test_successful_hypothesis_write_does_not_raise(self, dlq):
        driver = OkDriver()
        writer = ResilientNeo4jWriter(driver, dlq)
        hyp    = make_hypothesis()
        await writer.write_hypothesis(hyp)   # should not raise

    @pytest.mark.asyncio
    async def test_failed_hypothesis_write_routes_to_dlq(self, dlq):
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(driver, dlq, retry_delays=(0.0,))
        hyp    = make_hypothesis(session_id="sess-soft-fail")
        await writer.write_hypothesis(hyp)   # must NOT raise
        assert len(dlq.published) == 1
        assert dlq.published[0]["operation"] == "write_hypothesis"

    @pytest.mark.asyncio
    async def test_failed_hypothesis_write_does_not_raise(self, dlq):
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(driver, dlq, retry_delays=(0.0,))
        hyp    = make_hypothesis()
        # This is the key contract: SOFT writes never block inference
        await writer.write_hypothesis(hyp)

    @pytest.mark.asyncio
    async def test_dlq_message_contains_session_id(self, dlq):
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(driver, dlq, retry_delays=(0.0,))
        hyp    = make_hypothesis(session_id="sess-dlq-check")
        await writer.write_hypothesis(hyp)
        assert dlq.published[0]["session_id"] == "sess-dlq-check"

    @pytest.mark.asyncio
    async def test_hypothesis_write_emits_written_signal_on_success(self, dlq, mocker):
        mock_emit = mocker.patch("ts_agent.resilience.graph_writer.eamgp.emit")
        driver    = OkDriver()
        writer    = ResilientNeo4jWriter(driver, dlq)
        await writer.write_hypothesis(make_hypothesis())
        signals   = [call.args[0] for call in mock_emit.call_args_list]
        assert "SEG_HYPOTHESIS_WRITTEN" in signals

    @pytest.mark.asyncio
    async def test_hypothesis_write_emits_fail_signal_on_error(self, dlq, mocker):
        mock_emit = mocker.patch("ts_agent.resilience.graph_writer.eamgp.emit")
        driver    = FailingDriver()
        writer    = ResilientNeo4jWriter(driver, dlq, retry_delays=(0.0,))
        await writer.write_hypothesis(make_hypothesis())
        signals   = [call.args[0] for call in mock_emit.call_args_list]
        assert "SEG_HYPOTHESIS_WRITE_FAIL" in signals

    @pytest.mark.asyncio
    async def test_hypothesis_write_circuit_state_unaffected_by_soft_failure(self, dlq):
        """Soft write failures must NOT trip the circuit breaker."""
        driver = FailingDriver()
        writer = ResilientNeo4jWriter(driver, dlq, retry_delays=(0.0,), circuit_threshold=1)
        await writer.write_hypothesis(make_hypothesis())
        # Circuit breaker should remain CLOSED — soft writes don't interact with it
        assert writer.circuit_state == CircuitState.CLOSED
