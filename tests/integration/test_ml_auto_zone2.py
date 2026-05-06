"""
Integration test for ML-automatic Zone 2 gap-fill conversation.

This test simulates a complete consumer conversation with automatic ML prediction
happening after each answer, demonstrating:
1. Questions prioritized by SHAP/ML
2. Prediction confidence improving with each answer
3. Early stopping when confidence reaches threshold
4. Full logging for explainability
"""
import pytest
import json
from unittest.mock import Mock, patch
from ts_agent.domain.models import TraitGraph, NodeState
from ts_agent.zones.zone2.tools_ml_auto import (
    record_consumer_answer_ml_auto,
    get_next_question_ml_prioritized,
    check_graph_completeness_ml_aware,
    match_segment,
    STATE_GRAPH,
    STATE_TURN,
    STATE_HYPOTHESIS,
    STATE_PREDICTION_HISTORY,
    STATE_PREV_CONFIDENCE,
)
from ts_agent.zones.zone2.tools import _graph_to_dict


@pytest.fixture
def investment_scenario_graph():
    """Create a realistic investment scenario graph."""
    graph = TraitGraph(
        session_id="integ-test-001",
        party_ref="PARTY-TEST",
        intent_id="INT-INVEST",
        situation_id="SIT-INVEST",
    )
    
    # Add typical investment-related traits
    graph.add_node(
        node_id="N1", char_id="CHAR-F2A-I1",
        branch="financial", label="Monthly Surplus",
        op=">=", target_value=250,
        fill_priority=1,
        fill_question_key="monthly_surplus",
    )
    graph.add_node(
        node_id="N2", char_id="CHAR-F2B-I1",
        branch="financial", label="Cash/Deposits",
        op=">=", target_value=5000,
        fill_priority=2,
        fill_question_key="cash_deposits",
    )
    graph.add_node(
        node_id="N3", char_id="CHAR-F2I-I1",
        branch="product", label="Current Investments",
        op="==", target_value=False,
        fill_priority=3,
        fill_question_key="has_investments",
    )
    graph.add_node(
        node_id="N4", char_id="CHAR-P2A-I1",
        branch="personal", label="Age",
        op=">=", target_value=30,
        fill_priority=4,
        fill_question_key="age",
    )
    graph.add_node(
        node_id="N5", char_id="CHAR-P2B-I1",
        branch="personal", label="Risk Tolerance",
        op="==", target_value="MODERATE",
        fill_priority=5,
        fill_question_key="risk_tolerance",
    )
    
    return graph


@pytest.mark.asyncio
async def test_full_ml_auto_conversation_flow():
    """
    Simulate a full conversation with ML predicting after each answer.
    
    Conversation flow:
    1. Q1: monthly_surplus → Answer: £600 → ML confidence: 0.45
    2. Q2: cash_deposits → Answer: £12,000 → ML confidence: 0.68
    3. Q3: has_investments → Answer: No → ML confidence: 0.83 (HIGH!)
    4. STOP - confidence threshold reached, proceed to match
    """
    # Setup
    graph = TraitGraph(
        session_id="test-conv-001",
        party_ref="PARTY001",
        intent_id="INT-INVEST",
        situation_id="SIT-INVEST",
    )
    graph.add_node(
        node_id="N1", char_id="CHAR-F2A-I1",
        branch="financial", label="monthly_surplus",
        op=">=", target_value=250, fill_priority=1,
        fill_question_key="monthly_surplus",
    )
    graph.add_node(
        node_id="N2", char_id="CHAR-F2B-I1",
        branch="financial", label="cash_deposits",
        op=">=", target_value=5000, fill_priority=2,
        fill_question_key="cash_deposits",
    )
    graph.add_node(
        node_id="N3", char_id="CHAR-F2I-I1",
        branch="product", label="current_investments",
        op="==", target_value=False, fill_priority=3,
        fill_question_key="has_investments",
    )
    
    mock_context = Mock()
    mock_context.state = {
        STATE_GRAPH: _graph_to_dict(graph),
        STATE_TURN: 0,
        STATE_PREDICTION_HISTORY: "[]",
    }
    
    # Mock ML predictor with evolving confidence
    with patch("ts_agent.zones.zone2.tools_ml_auto.SegmentPredictor") as MockPredictor:
        predictor_instance = MockPredictor.return_value
        
        # Turn 1: Answer monthly_surplus
        hypothesis1 = Mock()
        hypothesis1.top_segment_id = "SEG-INV-002"
        hypothesis1.top_confidence = 0.45  # Low confidence
        hypothesis1.model_version = "demo-1.3"
        hypothesis1.model_algorithm = "LR"
        hypothesis1.all_scores = {"SEG-INV-002": 0.45, "SEG-INV-001": 0.35}
        hypothesis1.shap_features = [
            {"f": "cash_deposits", "v": 0.52},  # Most important next
            {"f": "monthly_surplus", "v": 0.38},
        ]
        predictor_instance.predict.return_value = hypothesis1
        
        result1 = await record_consumer_answer_ml_auto(
            char_id="CHAR-F2A-I1",
            value="600",
            tool_context=mock_context,
        )
        
        # Verify Turn 1
        assert result1["success"] is True
        assert result1["ml_prediction"]["confidence"] == 0.45
        assert result1["ml_prediction"]["confidence_delta"] == 0.45
        assert result1["ml_prediction"]["high_confidence"] is False
        assert result1["ready_for_match"] is False
        print(f"✓ Turn 1: Confidence = 0.45, Not ready")
        
        # Turn 2: Answer cash_deposits
        hypothesis2 = Mock()
        hypothesis2.top_segment_id = "SEG-INV-002"
        hypothesis2.top_confidence = 0.68  # Improving
        hypothesis2.model_version = "demo-1.3"
        hypothesis2.model_algorithm = "LR"
        hypothesis2.all_scores = {"SEG-INV-002": 0.68, "SEG-INV-001": 0.22}
        hypothesis2.shap_features = [
            {"f": "current_investments", "v": 0.61},  # Most important now
            {"f": "cash_deposits", "v": 0.45},
        ]
        predictor_instance.predict.return_value = hypothesis2
        
        result2 = await record_consumer_answer_ml_auto(
            char_id="CHAR-F2B-I1",
            value="12000",
            tool_context=mock_context,
        )
        
        # Verify Turn 2
        assert result2["success"] is True
        assert result2["ml_prediction"]["confidence"] == 0.68
        assert result2["ml_prediction"]["confidence_delta"] == pytest.approx(0.23, abs=0.01)
        assert result2["ml_prediction"]["high_confidence"] is False
        assert result2["ready_for_match"] is False
        print(f"✓ Turn 2: Confidence = 0.68 (Δ+0.23), Not ready")
        
        # Turn 3: Answer has_investments
        hypothesis3 = Mock()
        hypothesis3.top_segment_id = "SEG-INV-002"
        hypothesis3.top_confidence = 0.83  # HIGH CONFIDENCE!
        hypothesis3.model_version = "demo-1.3"
        hypothesis3.model_algorithm = "LR"
        hypothesis3.all_scores = {"SEG-INV-002": 0.83, "SEG-INV-001": 0.12}
        hypothesis3.shap_features = [
            {"f": "current_investments", "v": 0.71},
            {"f": "cash_deposits", "v": 0.42},
        ]
        predictor_instance.predict.return_value = hypothesis3
        
        result3 = await record_consumer_answer_ml_auto(
            char_id="CHAR-F2I-I1",
            value="no",
            tool_context=mock_context,
        )
        
        # Verify Turn 3 - Should be ready!
        assert result3["success"] is True
        assert result3["ml_prediction"]["confidence"] == 0.83
        assert result3["ml_prediction"]["confidence_delta"] == pytest.approx(0.15, abs=0.01)
        assert result3["ml_prediction"]["high_confidence"] is True
        assert result3["ready_for_match"] is True  # READY TO MATCH!
        print(f"✓ Turn 3: Confidence = 0.83 (Δ+0.15), READY!")
        
        # Verify prediction history
        history_json = mock_context.state[STATE_PREDICTION_HISTORY]
        history = json.loads(history_json)
        assert len(history) == 3
        assert history[0]["top_confidence"] == 0.45
        assert history[1]["top_confidence"] == 0.68
        assert history[2]["top_confidence"] == 0.83
        print(f"✓ Prediction history logged: {len(history)} entries")
        
        # Verify we only asked 3 questions (not all 5!)
        assert mock_context.state[STATE_TURN] == 3
        print(f"✓ Efficient: Only 3 questions asked vs 5 total traits")


@pytest.mark.asyncio
async def test_question_prioritization_by_shap():
    """Test that questions are prioritized by SHAP importance, not just fill_priority."""
    graph = TraitGraph(
        session_id="test-priority-001",
        party_ref="PARTY001",
        intent_id="INT-INVEST",
        situation_id="SIT-INVEST",
    )
    
    # Node with high fill_priority (1 = highest)
    graph.add_node(
        node_id="N1", char_id="CHAR-F2A-I1",
        branch="financial", label="monthly_surplus",
        op=">=", target_value=250,
        fill_priority=1,  # High priority
        fill_question_key="monthly_surplus",
    )
    
    # Node with lower fill_priority but HIGH SHAP importance
    graph.add_node(
        node_id="N2", char_id="CHAR-F2B-I1",
        branch="financial", label="cash_deposits",
        op=">=", target_value=5000,
        fill_priority=5,  # Low priority
        fill_question_key="cash_deposits",
    )
    
    mock_context = Mock()
    mock_context.state = {
        STATE_GRAPH: _graph_to_dict(graph),
        STATE_HYPOTHESIS: json.dumps({
            "top_segment_id": "SEG-INV-002",
            "top_confidence": 0.55,
            "shap_features": [
                {"f": "cash_deposits", "v": 0.82},  # HIGH SHAP!
                {"f": "monthly_surplus", "v": 0.24},  # Low SHAP
            ],
        }),
    }
    
    result = await get_next_question_ml_prioritized(mock_context)
    
    # Should suggest cash_deposits (high SHAP) despite low fill_priority
    assert result["char_id"] == "CHAR-F2B-I1"
    assert "SHAP" in result["priority_reason"]
    print(f"✓ ML prioritization: Chose CHAR-F2B-I1 based on SHAP, not fill_priority")


@pytest.mark.asyncio
async def test_early_stopping_vs_traditional():
    """
    Compare ML early stopping vs traditional 90% completeness.
    
    Scenario: 10 traits, but ML reaches high confidence after just 4 answers.
    Traditional mode would ask 9 questions (90%), ML mode stops at 4.
    """
    graph = TraitGraph(
        session_id="test-early-stop",
        party_ref="PARTY001",
        intent_id="INT-INVEST",
        situation_id="SIT-INVEST",
    )
    
    # Add 10 traits
    for i in range(10):
        graph.add_node(
            node_id=f"N{i}",
            char_id=f"CHAR-TEST-{i}",
            branch="test",
            label=f"trait_{i}",
            op=">=",
            target_value=100,
            fill_priority=i+1,
            fill_question_key=f"trait_{i}",
        )
    
    # Mark 4 as KNOWN (40% completeness)
    for i in range(4):
        graph.nodes[f"N{i}"].state = NodeState.KNOWN
        graph.nodes[f"N{i}"].value = 100
    
    mock_context = Mock()
    mock_context.state = {
        STATE_GRAPH: _graph_to_dict(graph),
        STATE_HYPOTHESIS: json.dumps({
            "top_confidence": 0.85,  # High ML confidence!
        }),
    }
    
    result = await check_graph_completeness_ml_aware(mock_context)
    
    # Verify early stopping
    assert result["completeness"] == 0.4  # Only 40%
    assert result["ml_confidence"] == 0.85
    assert result["ml_ready"] is True
    assert result["traditional_ready"] is False  # <90%
    assert result["ready_for_match"] is True  # ML wins!
    
    print(f"✓ Early stopping: Ready at 40% completeness due to ML confidence")
    print(f"  Traditional would require 90% = 9 questions")
    print(f"  ML stopped at 40% = 4 questions (56% reduction)")


@pytest.mark.asyncio
async def test_explainability_logging():
    """Verify that all predictions are logged for regulatory explainability."""
    graph = TraitGraph(
        session_id="test-explainability",
        party_ref="PARTY001",
        intent_id="INT-INVEST",
        situation_id="SIT-INVEST",
    )
    graph.add_node(
        node_id="N1", char_id="CHAR-F2A-I1",
        branch="financial", label="monthly_surplus",
        op=">=", target_value=250, fill_priority=1,
        fill_question_key="monthly_surplus",
    )
    
    mock_context = Mock()
    mock_context.state = {
        STATE_GRAPH: _graph_to_dict(graph),
        STATE_TURN: 0,
        STATE_PREDICTION_HISTORY: "[]",
    }
    
    with patch("ts_agent.zones.zone2.tools_ml_auto.SegmentPredictor") as MockPredictor:
        hypothesis = Mock()
        hypothesis.top_segment_id = "SEG-INV-002"
        hypothesis.top_confidence = 0.65
        hypothesis.model_version = "demo-1.3"
        hypothesis.model_algorithm = "LR"
        hypothesis.all_scores = {"SEG-INV-002": 0.65}
        hypothesis.shap_features = [
            {"f": "monthly_surplus", "v": 0.42},
            {"f": "cash_deposits", "v": 0.31},
        ]
        MockPredictor.return_value.predict.return_value = hypothesis
        
        await record_consumer_answer_ml_auto(
            char_id="CHAR-F2A-I1",
            value="500",
            tool_context=mock_context,
        )
    
    # Verify comprehensive logging
    history_json = mock_context.state[STATE_PREDICTION_HISTORY]
    history = json.loads(history_json)
    
    assert len(history) == 1
    entry = history[0]
    
    # Must have all explainability fields
    assert "turn" in entry
    assert "top_segment_id" in entry
    assert "top_confidence" in entry
    assert "confidence_delta" in entry
    assert "all_scores" in entry  # All segment probabilities
    assert "shap_features" in entry  # Feature importance
    assert "known_trait_count" in entry
    
    print(f"✓ Explainability: Full prediction logged with SHAP features")
    print(f"  Top segment: {entry['top_segment_id']}")
    print(f"  Confidence: {entry['top_confidence']}")
    print(f"  SHAP features: {len(entry['shap_features'])} recorded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
