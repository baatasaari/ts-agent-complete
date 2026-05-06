"""
tests/unit/zones/test_zone1_graph_builder.py
=============================================
Unit tests for ts_agent.zones.zone1_graph_builder

Mock writer is injected — no Neo4j required.
"""

import pytest

from ts_agent.domain.models import EdgeType, NodeState
from ts_agent.zones.zone1_graph_builder import (
    ConflictPair,
    TraitGraphBuilder,
)
from tests.fixtures.factories import (
    make_intent,
    make_situation_config,
    make_trait_config,
    make_user_profile,
)
from ts_agent.domain.models import NodeBranch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / stubs
# ──────────────────────────────────────────────────────────────────────────────

class MockWriter:
    """In-memory writer stub — records calls without any Neo4j dependency."""

    def __init__(self, should_raise: Exception | None = None) -> None:
        self.calls: list = []
        self._raise = should_raise

    async def write_hard(self, graph):
        self.calls.append(graph)
        if self._raise:
            raise self._raise


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTraitGraphBuilder:

    @pytest.fixture()
    def writer(self) -> MockWriter:
        return MockWriter()

    @pytest.fixture()
    def builder(self, writer) -> TraitGraphBuilder:
        config = make_situation_config()
        return TraitGraphBuilder(config=config, writer=writer)

    # ── Initialisation ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_build_creates_one_node_per_config_entry(self, builder, writer):
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1"),
            make_trait_config("CHAR-F2A-I1", NodeBranch.FINANCIAL),
        ])
        b = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(make_intent(), make_user_profile(bank_data={}))
        assert len(graph.nodes) == 2

    @pytest.mark.asyncio
    async def test_build_calls_writer_once(self, builder, writer):
        await builder.build(make_intent(), make_user_profile())
        assert len(writer.calls) == 1

    @pytest.mark.asyncio
    async def test_build_propagates_party_ref(self, builder):
        graph = await builder.build(
            make_intent(), make_user_profile(party_ref="PARTY-XYZ")
        )
        assert graph.party_ref == "PARTY-XYZ"

    @pytest.mark.asyncio
    async def test_build_sets_intent_id(self, builder):
        graph = await builder.build(
            make_intent(intent_id="INTENT-ISA"), make_user_profile()
        )
        assert graph.intent_id == "INTENT-ISA"

    # ── Bank data population ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_bank_data_populates_known_nodes(self, writer):
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1"),
            make_trait_config("CHAR-F2A-I1", NodeBranch.FINANCIAL),
        ])
        b = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(
            make_intent(),
            make_user_profile(bank_data={"CHAR-P1A-I1": 3, "CHAR-F2A-I1": 750.0}),
        )
        assert graph.node_by_char_id("CHAR-P1A-I1").state == NodeState.KNOWN
        assert graph.node_by_char_id("CHAR-F2A-I1").state == NodeState.KNOWN

    @pytest.mark.asyncio
    async def test_missing_bank_data_leaves_node_missing(self, builder):
        profile = make_user_profile(bank_data={})   # no data at all
        graph   = await builder.build(make_intent(), profile)
        for node in graph.nodes.values():
            assert node.state == NodeState.MISSING

    @pytest.mark.asyncio
    async def test_known_node_value_matches_bank_data(self, writer):
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1"),
        ])
        b = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(make_intent(), make_user_profile(bank_data={"CHAR-P1A-I1": 42}))
        assert graph.node_by_char_id("CHAR-P1A-I1").value == 42

    # ── Mode exclusions ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_email_test_nodes_excluded_on_mobile_channel(self, writer):
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1"),
            make_trait_config("CHAR-EMAIL-01", email_test_node=True),
        ])
        b = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(
            make_intent(channel="mobile"), make_user_profile(bank_data={})
        )
        email_node = graph.node_by_char_id("CHAR-EMAIL-01")
        assert email_node.state == NodeState.EXCLUDED

    @pytest.mark.asyncio
    async def test_email_test_nodes_not_excluded_on_email_channel(self, writer):
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1"),
            make_trait_config("CHAR-EMAIL-01", email_test_node=True),
        ])
        b = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(
            make_intent(channel="email"), make_user_profile(bank_data={})
        )
        email_node = graph.node_by_char_id("CHAR-EMAIL-01")
        # Should be MISSING (no bank data), not EXCLUDED
        assert email_node.state == NodeState.MISSING

    # ── Dependency edges ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_depends_on_edges_built_between_different_priorities(self, writer):
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1", fill_priority=1),
            make_trait_config("CHAR-F2A-I1", NodeBranch.FINANCIAL, fill_priority=2),
            make_trait_config("CHAR-F2B-I1", NodeBranch.FINANCIAL, fill_priority=3),
        ])
        b     = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(make_intent(), make_user_profile(bank_data={}))

        dep_edges = [e for e in graph.edges if e.edge_type == EdgeType.DEPENDS_ON]
        assert len(dep_edges) >= 1

    @pytest.mark.asyncio
    async def test_no_depends_on_edges_when_single_priority(self, writer):
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1", fill_priority=1),
            make_trait_config("CHAR-F2A-I1", NodeBranch.FINANCIAL, fill_priority=1),
        ])
        b     = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(make_intent(), make_user_profile(bank_data={}))

        dep_edges = [e for e in graph.edges if e.edge_type == EdgeType.DEPENDS_ON]
        assert len(dep_edges) == 0

    @pytest.mark.asyncio
    async def test_known_nodes_excluded_from_dependency_edges(self, writer):
        """KNOWN nodes should not appear in DEPENDS_ON edges."""
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1", fill_priority=1),
            make_trait_config("CHAR-F2A-I1", NodeBranch.FINANCIAL, fill_priority=2),
        ])
        b     = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(
            make_intent(),
            make_user_profile(bank_data={"CHAR-P1A-I1": 3}),   # P1A is KNOWN
        )
        dep_edges = [e for e in graph.edges if e.edge_type == EdgeType.DEPENDS_ON]
        # CHAR-P1A-I1 is KNOWN so it should not be in fill ordering edges
        known_node_id = graph.node_by_char_id("CHAR-P1A-I1").node_id
        for edge in dep_edges:
            assert edge.from_id != known_node_id
            assert edge.to_id   != known_node_id

    # ── Conflict edges ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_conflict_edges_built_for_configured_pairs(self, writer):
        pair   = ConflictPair("CHAR-P1A-I1", "CHAR-F2A-I1", ("SEG-A", "SEG-B"))
        config = make_situation_config(
            trait_configs=[
                make_trait_config("CHAR-P1A-I1"),
                make_trait_config("CHAR-F2A-I1", NodeBranch.FINANCIAL),
            ],
            conflict_pairs=[pair],
        )
        b     = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(make_intent(), make_user_profile(bank_data={}))

        conflict_edges = [e for e in graph.edges if e.edge_type == EdgeType.CONFLICTS_WITH]
        assert len(conflict_edges) == 1
        assert conflict_edges[0].properties["reason"] == "mutual_exclusion"

    @pytest.mark.asyncio
    async def test_conflict_edge_not_built_if_char_id_missing(self, writer):
        """Conflict pair references a char_id not in the graph → silently skip."""
        pair   = ConflictPair("CHAR-NONEXISTENT", "CHAR-P1A-I1", ())
        config = make_situation_config(
            trait_configs=[make_trait_config("CHAR-P1A-I1")],
            conflict_pairs=[pair],
        )
        b     = TraitGraphBuilder(config=config, writer=writer)
        graph = await b.build(make_intent(), make_user_profile(bank_data={}))

        conflict_edges = [e for e in graph.edges if e.edge_type == EdgeType.CONFLICTS_WITH]
        assert len(conflict_edges) == 0

    # ── Writer failure ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_build_propagates_writer_exception(self):
        writer = MockWriter(should_raise=RuntimeError("Neo4j down"))
        config = make_situation_config()
        b      = TraitGraphBuilder(config=config, writer=writer)
        with pytest.raises(RuntimeError, match="Neo4j down"):
            await b.build(make_intent(), make_user_profile())

    # ── EAMGP signals ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_graph_build_complete_signal_emitted(self, mocker, builder):
        mock_emit = mocker.patch("ts_agent.zones.zone1_graph_builder.eamgp.emit")
        await builder.build(make_intent(), make_user_profile())
        signals = [call.args[0] for call in mock_emit.call_args_list]
        assert "GRAPH_BUILD_COMPLETE" in signals

    @pytest.mark.asyncio
    async def test_graph_build_start_signal_emitted(self, mocker, builder):
        mock_emit = mocker.patch("ts_agent.zones.zone1_graph_builder.eamgp.emit")
        await builder.build(make_intent(), make_user_profile())
        signals = [call.args[0] for call in mock_emit.call_args_list]
        assert "GRAPH_BUILD_START" in signals

    @pytest.mark.asyncio
    async def test_node_excluded_signal_emitted_for_email_node(self, mocker, writer):
        mock_emit = mocker.patch("ts_agent.zones.zone1_graph_builder.eamgp.emit")
        config = make_situation_config(trait_configs=[
            make_trait_config("CHAR-P1A-I1"),
            make_trait_config("CHAR-EMAIL-01", email_test_node=True),
        ])
        b = TraitGraphBuilder(config=config, writer=writer)
        await b.build(make_intent(channel="mobile"), make_user_profile(bank_data={}))
        signals = [call.args[0] for call in mock_emit.call_args_list]
        assert "GRAPH_NODE_EXCLUDED" in signals
