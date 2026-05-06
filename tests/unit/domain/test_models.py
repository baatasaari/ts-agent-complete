"""
tests/unit/domain/test_models.py
=================================
Unit tests for ts_agent.domain.models

All tests are pure in-memory — no I/O, no external dependencies.
"""

import pytest

from ts_agent.domain.models import (
    EdgeType,
    GraphEdge,
    GateDisposition,
    HypothesisDisposition,
    ModelAlgorithm,
    NodeBranch,
    NodeState,
    SegmentHypothesis,
    SegmentRank,
    ShapFeature,
    TraitGraph,
    TraitGraphIncompleteError,
    TraitNode,
)
from tests.fixtures.factories import (
    make_complete_graph,
    make_excluded_node,
    make_hypothesis,
    make_incomplete_graph,
    make_known_node,
    make_missing_node,
    make_trait_node,
)


# ──────────────────────────────────────────────────────────────────────────────
# TraitNode
# ──────────────────────────────────────────────────────────────────────────────

class TestTraitNode:

    def test_default_state_is_missing(self):
        node = make_trait_node()
        assert node.state == NodeState.MISSING

    def test_with_state_returns_new_node(self):
        original = make_trait_node()
        updated  = original.with_state(NodeState.KNOWN, value=42, populated_source="OCIS")

        assert updated.state == NodeState.KNOWN
        assert updated.value == 42
        assert updated.populated_source == "OCIS"
        # original is unchanged (frozen dataclass)
        assert original.state == NodeState.MISSING
        assert original.value is None

    def test_with_state_preserves_all_other_fields(self):
        original = make_trait_node(char_id="CHAR-X", fill_priority=7)
        updated  = original.with_state(NodeState.EXCLUDED)

        assert updated.char_id      == "CHAR-X"
        assert updated.fill_priority == 7

    def test_value_hash_is_none_when_value_is_none(self):
        node = make_trait_node()
        assert node.value_hash() is None

    def test_value_hash_returns_hex_string(self):
        node = make_known_node(value="hello")
        h = node.value_hash()
        assert isinstance(h, str)
        assert len(h) == 64        # SHA-256 hex

    def test_value_hash_is_deterministic(self):
        node = make_known_node(value=42)
        assert node.value_hash() == node.value_hash()

    def test_value_hash_differs_for_different_values(self):
        n1 = make_known_node(value=1)
        n2 = make_known_node(value=2)
        assert n1.value_hash() != n2.value_hash()

    def test_frozen_prevents_direct_mutation(self):
        node = make_trait_node()
        with pytest.raises((AttributeError, TypeError)):
            node.state = NodeState.KNOWN  # type: ignore[misc]

    def test_node_branch_enum_values(self):
        assert NodeBranch.PERSONAL.value    == "PERSONAL"
        assert NodeBranch.FINANCIAL.value   == "FINANCIAL"
        assert NodeBranch.PRODUCT.value     == "PRODUCT"
        assert NodeBranch.BEHAVIOURAL.value == "BEHAVIOURAL"


# ──────────────────────────────────────────────────────────────────────────────
# TraitGraph
# ──────────────────────────────────────────────────────────────────────────────

class TestTraitGraph:

    def test_add_node_creates_has_trait_edge(self):
        g = TraitGraph(session_id="sess-1")
        node = make_trait_node()
        g.add_node(node)

        has_trait = [e for e in g.edges if e.edge_type == EdgeType.HAS_TRAIT]
        assert len(has_trait) == 1
        assert has_trait[0].from_id == "sess-1"
        assert has_trait[0].to_id   == node.node_id

    def test_add_multiple_nodes(self):
        g = TraitGraph()
        g.add_node(make_known_node("CHAR-A"))
        g.add_node(make_known_node("CHAR-B"))
        assert len(g.nodes) == 2

    def test_is_complete_true_when_no_missing(self):
        g = make_complete_graph()
        assert g.is_complete() is True

    def test_is_complete_false_with_missing_node(self):
        g = make_incomplete_graph()
        assert g.is_complete() is False

    def test_is_complete_true_with_only_known_and_excluded(self):
        g = TraitGraph()
        g.add_node(make_known_node("CHAR-A"))
        g.add_node(make_excluded_node("CHAR-B"))
        assert g.is_complete() is True

    def test_freeze_raises_if_not_complete(self):
        g = make_incomplete_graph()
        with pytest.raises(TraitGraphIncompleteError):
            g.freeze()

    def test_freeze_succeeds_when_complete(self):
        g = make_complete_graph()
        g.freeze()
        assert g.is_frozen is True

    def test_frozen_graph_blocks_add_node(self):
        g = make_complete_graph()
        g.freeze()
        with pytest.raises(RuntimeError, match="Cannot mutate"):
            g.add_node(make_known_node("CHAR-NEW"))

    def test_frozen_graph_blocks_update_node(self):
        g = make_complete_graph()
        g.freeze()
        first_node = next(iter(g.nodes.values()))
        with pytest.raises(RuntimeError, match="Cannot mutate"):
            g.update_node(first_node.with_state(NodeState.EXCLUDED))

    def test_frozen_graph_blocks_add_edge(self):
        g = make_complete_graph()
        g.freeze()
        edge = GraphEdge(from_id="a", to_id="b", edge_type=EdgeType.DEPENDS_ON)
        with pytest.raises(RuntimeError, match="Cannot mutate"):
            g.add_edge(edge)

    def test_update_node_replaces_existing(self):
        g = TraitGraph()
        node = make_missing_node("CHAR-F2A-I1")
        g.add_node(node)
        known = node.with_state(NodeState.KNOWN, value=999.0, populated_source="CBS")
        g.update_node(known)

        stored = g.nodes[node.node_id]
        assert stored.state == NodeState.KNOWN
        assert stored.value == 999.0

    def test_update_node_raises_for_unknown_id(self):
        g = TraitGraph()
        ghost = make_known_node()
        with pytest.raises(KeyError):
            g.update_node(ghost)

    def test_node_by_char_id_returns_correct_node(self):
        g = make_complete_graph()
        n = g.node_by_char_id("CHAR-P1A-I1")
        assert n is not None
        assert n.char_id == "CHAR-P1A-I1"

    def test_node_by_char_id_returns_none_if_not_found(self):
        g = make_complete_graph()
        assert g.node_by_char_id("CHAR-DOES-NOT-EXIST") is None

    def test_missing_nodes_returns_only_missing(self):
        g = make_incomplete_graph()
        missing = g.missing_nodes()
        assert all(n.state == NodeState.MISSING for n in missing)
        assert len(missing) >= 1

    def test_known_nodes_returns_only_known(self):
        g = make_complete_graph()
        known = g.known_nodes()
        assert all(n.state == NodeState.KNOWN for n in known)

    def test_excluded_nodes_returns_only_excluded(self):
        g = make_complete_graph()
        excluded = g.excluded_nodes()
        assert all(n.state == NodeState.EXCLUDED for n in excluded)

    def test_stats_sums_to_total(self):
        g = make_complete_graph()
        s = g.stats()
        assert s["known"] + s["missing"] + s["excluded"] == s["total"]

    def test_stats_edge_count_matches_edges_list(self):
        g = make_complete_graph()
        assert g.stats()["edges"] == len(g.edges)

    def test_freeze_error_message_lists_missing_char_ids(self):
        g = make_incomplete_graph()
        with pytest.raises(TraitGraphIncompleteError, match="CHAR-F2A-I1"):
            g.freeze()


# ──────────────────────────────────────────────────────────────────────────────
# SegmentHypothesis
# ──────────────────────────────────────────────────────────────────────────────

class TestSegmentHypothesis:

    def test_top_segment_id_returns_first_ranked(self):
        hyp = make_hypothesis(top_segment_id="SEG-I1-A", top_confidence=0.85)
        assert hyp.top_segment_id == "SEG-I1-A"

    def test_top_confidence_returns_first_probability(self):
        hyp = make_hypothesis(top_confidence=0.85)
        assert abs(hyp.top_confidence - 0.85) < 1e-9

    def test_top_segment_id_none_when_no_ranked_segments(self):
        hyp = SegmentHypothesis(
            session_id="s", turn=1,
            model_version="v", model_algorithm=ModelAlgorithm.LOGISTIC_REGRESSION,
            known_trait_count=0,
        )
        assert hyp.top_segment_id is None
        assert hyp.top_confidence == 0.0

    def test_is_undecidable_true_for_undecidable(self):
        from ts_agent.domain.models import HypothesisDisposition
        hyp = SegmentHypothesis(
            session_id="s", turn=0,
            model_version="v", model_algorithm=ModelAlgorithm.LOGISTIC_REGRESSION,
            known_trait_count=3,
            disposition=HypothesisDisposition.UNDECIDABLE,
        )
        assert hyp.is_undecidable() is True

    def test_is_undecidable_false_for_active(self):
        hyp = make_hypothesis()
        assert hyp.is_undecidable() is False

    def test_to_neo4j_params_contains_required_keys(self):
        hyp = make_hypothesis()
        params = hyp.to_neo4j_params()
        required = {
            "hypothesis_id", "session_id", "turn",
            "top_segment_id", "top_confidence", "shap_json",
            "model_version", "model_algorithm", "known_trait_count",
            "disposition",
        }
        assert required.issubset(params.keys())

    def test_to_neo4j_params_shap_json_is_valid_json(self):
        import json
        hyp = make_hypothesis()
        params = hyp.to_neo4j_params()
        parsed = json.loads(params["shap_json"])
        assert isinstance(parsed, list)
        assert parsed[0]["feature"] == "monthly_surplus"

    def test_hypothesis_id_is_unique_per_instance(self):
        h1 = make_hypothesis()
        h2 = make_hypothesis()
        assert h1.hypothesis_id != h2.hypothesis_id
