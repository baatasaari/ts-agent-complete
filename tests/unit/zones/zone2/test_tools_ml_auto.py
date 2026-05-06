"""
Unit tests for ML-automatic gap-fill tools.
"""
import json
import pytest
from unittest.mock import Mock, patch

from ts_agent.domain.models import GapFillStrategy, NodeState, TraitGraph
from ts_agent.zones.zone2.tools_ml_auto import (
    record_consumer_answer_ml_auto,
    get_next_question_ml_prioritized,
    check_graph_completeness_ml_aware,
    STATE_GRAPH,
    STATE_TURN,
    STATE_HYPOTHESIS,
    STATE_PREV_CONFIDENCE,
    STATE_PREDICTION_HISTORY,
)
from ts_agent.zones.zone2.tools import _graph_to_dict


@pytest.fixture
def mock_tool_context():
    """Mock ADK ToolContext for testing."""
    ctx = Mock()
    ctx.state = {}
    return ctx


@pytest.fixture
def sample_graph():
    """Create a sample TraitGraph for testing."""
    graph = TraitGraph(
        session_id="test-session-123",
        party_ref="PARTY001",
        intent_id="INT-SAVE",
        situation_id="SIT-BUDGET",
    )
    graph.add_node(
        node_id="N1", char_id="CHAR-F2A-I1",
        branch="income", label="Monthly Surplus",
        op=">=", target_value=100,
        fill_priority=1,
        fill_question_key="monthly_surplus",
    )
    graph.add_node(
        node_id="N2", char_id="CHAR-F2B-I1",
        branch="income", label="Cash/Deposits",
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
    return graph


@pytest.mark.asyncio
async def test_record_answer_triggers_ml_prediction(mock_tool_context, sample_graph):
    """Test that recording an answer automatically triggers ML prediction."""
    # Setup
    mock_tool_context.state[STATE_GRAPH] = _graph_to_dict(sample_graph)
    mock_tool_context.state[STATE_TURN] = 0
    
    # Mock ML predictor
    with patch("ts_agent.zones.zone2.tools_ml_auto.SegmentPredictor") as MockPredictor:
        mock_hypothesis = Mock()
        mock_hypothesis.top_segment_id = "SEG-INV-002"
        mock_hypothesis.top_confidence = 0.65
        mock_hypothesis.model_version = "demo-1.3"
        mock_hypothesis.model_algorithm = "LR"
        mock_hypothesis.all_scores = {
            "SEG-INV-002": 0.65,
            "SEG-INV-001": 0.25,
            "SEG-SD-001": 0.10,
        }
        mock_hypothesis.shap_features = [
            {"f": "monthly_surplus", "v": 0.42},
            {"f": "cash_deposits", "v": 0.31},
        ]
        
        MockPredictor.return_value.predict.return_value = mock_hypothesis
        
        # Execute
        result = await record_consumer_answer_ml_auto(
            char_id="CHAR-F2A-I1",
            value="500",
            tool_context=mock_tool_context,
        )
    
    # Verify
    assert result["success"] is True
    assert result["char_id"] == "CHAR-F2A-I1"
    assert "ml_prediction" in result
    assert result["ml_prediction"]["segment_id"] == "SEG-INV-002"
    assert result["ml_prediction"]["confidence"] == 0.65
    assert "confidence_delta" in result["ml_prediction"]
    assert "top_3_segments" in result
    
    # Verify prediction was logged to state
    assert STATE_HYPOTHESIS in mock_tool_context.state
    hypothesis_json = mock_tool_context.state[STATE_HYPOTHESIS]
    hypothesis = json.loads(hypothesis_json)
    assert hypothesis["top_segment_id"] == "SEG-INV-002"
    assert hypothesis["top_confidence"] == 0.65


@pytest.mark.asyncio
async def test_confidence_delta_tracking(mock_tool_context, sample_graph):
    """Test that confidence delta is correctly tracked across turns."""
    mock_tool_context.state[STATE_GRAPH] = _graph_to_dict(sample_graph)
    mock_tool_context.state[STATE_TURN] = 0
    mock_tool_context.state[STATE_PREV_CONFIDENCE] = 0.45  # Previous prediction
    
    with patch("ts_agent.zones.zone2.tools_ml_auto.SegmentPredictor") as MockPredictor:
        mock_hypothesis = Mock()
        mock_hypothesis.top_segment_id = "SEG-INV-002"
        mock_hypothesis.top_confidence = 0.78  # Improved!
        mock_hypothesis.model_version = "demo-1.3"
        mock_hypothesis.model_algorithm = "LR"
        mock_hypothesis.all_scores = {"SEG-INV-002": 0.78}
        mock_hypothesis.shap_features = []
        
        MockPredictor.return_value.predict.return_value = mock_hypothesis
        
        result = await record_consumer_answer_ml_auto(
            char_id="CHAR-F2A-I1",
            value="500",
            tool_context=mock_tool_context,
        )
    
    # Verify delta calculation
    assert result["ml_prediction"]["confidence_delta"] == pytest.approx(0.33, abs=0.01)
    assert mock_tool_context.state[STATE_PREV_CONFIDENCE] == 0.78


@pytest.mark.asyncio
async def test_prediction_history_logging(mock_tool_context, sample_graph):
    """Test that prediction history is maintained for explainability."""
    mock_tool_context.state[STATE_GRAPH] = _graph_to_dict(sample_graph)
    mock_tool_context.state[STATE_TURN] = 0
    mock_tool_context.state[STATE_PREDICTION_HISTORY] = json.dumps([])
    
    with patch("ts_agent.zones.zone2.tools_ml_auto.SegmentPredictor") as MockPredictor:
        mock_hypothesis = Mock()
        mock_hypothesis.top_segment_id = "SEG-INV-001"
        mock_hypothesis.top_confidence = 0.55
        mock_hypothesis.model_version = "demo-1.3"
        mock_hypothesis.model_algorithm = "LR"
        mock_hypothesis.all_scores = {"SEG-INV-001": 0.55}
        mock_hypothesis.shap_features = [{"f": "monthly_surplus", "v": 0.42}]
        
        MockPredictor.return_value.predict.return_value = mock_hypothesis
        
        await record_consumer_answer_ml_auto(
            char_id="CHAR-F2A-I1",
            value="500",
            tool_context=mock_tool_context,
        )
    
    # Verify history
    history_json = mock_tool_context.state[STATE_PREDICTION_HISTORY]
    history = json.loads(history_json)
    assert len(history) == 1
    assert history[0]["top_segment_id"] == "SEG-INV-001"
    assert history[0]["top_confidence"] == 0.55
    assert "shap_features" in history[0]


@pytest.mark.asyncio
async def test_ready_for_match_high_confidence(mock_tool_context, sample_graph):
    """Test that high ML confidence triggers ready_for_match."""
    mock_tool_context.state[STATE_GRAPH] = _graph_to_dict(sample_graph)
    mock_tool_context.state[STATE_TURN] = 0
    
    with patch("ts_agent.zones.zone2.tools_ml_auto.SegmentPredictor") as MockPredictor:
        mock_hypothesis = Mock()
        mock_hypothesis.top_segment_id = "SEG-INV-002"
        mock_hypothesis.top_confidence = 0.82  # Above 75% threshold!
        mock_hypothesis.model_version = "demo-1.3"
        mock_hypothesis.model_algorithm = "LR"
        mock_hypothesis.all_scores = {"SEG-INV-002": 0.82}
        mock_hypothesis.shap_features = []
        
        MockPredictor.return_value.predict.return_value = mock_hypothesis
        
        result = await record_consumer_answer_ml_auto(
            char_id="CHAR-F2A-I1",
            value="500",
            tool_context=mock_tool_context,
        )
    
    # Verify readiness
    assert result["ready_for_match"] is True
    assert result["ml_prediction"]["high_confidence"] is True


@pytest.mark.asyncio
async def test_get_next_question_ml_prioritized(mock_tool_context, sample_graph):
    """Test that questions are prioritized by SHAP importance."""
    # Setup with ML hypothesis
    mock_tool_context.state[STATE_GRAPH] = _graph_to_dict(sample_graph)
    mock_tool_context.state[STATE_HYPOTHESIS] = json.dumps({
        "top_segment_id": "SEG-INV-002",
        "top_confidence": 0.65,
        "shap_features": [
            {"f": "cash_deposits", "v": 0.52},  # Most important!
            {"f": "monthly_surplus", "v": 0.28},
        ],
    })
    
    result = await get_next_question_ml_prioritized(mock_tool_context)
    
    # Should suggest cash_deposits (highest SHAP) even though monthly_surplus has higher fill_priority
    assert result["done"] is False
    assert "CHAR-F2B-I1" in result["char_id"]  # cash_deposits
    assert "priority_reason" in result
    assert result["current_ml_confidence"] == 0.65


@pytest.mark.asyncio
async def test_check_completeness_ml_aware(mock_tool_context, sample_graph):
    """Test completeness check with ML confidence awareness."""
    # Mark one node as KNOWN
    sample_graph.nodes["N1"].state = NodeState.KNOWN
    sample_graph.nodes["N1"].value = 500
    
    mock_tool_context.state[STATE_GRAPH] = _graph_to_dict(sample_graph)
    mock_tool_context.state[STATE_HYPOTHESIS] = json.dumps({
        "top_confidence": 0.81,  # High confidence
    })
    
    result = await check_graph_completeness_ml_aware(mock_tool_context)
    
    # Should be ready due to ML confidence, despite low graph completeness
    assert result["ml_confidence"] == 0.81
    assert result["ml_ready"] is True
    assert result["ready_for_match"] is True
    assert result["completeness"] < 0.5  # Only 1/3 nodes known


@pytest.mark.asyncio
async def test_ml_prediction_failure_handling(mock_tool_context, sample_graph):
    """Test graceful handling of ML prediction failures."""
    mock_tool_context.state[STATE_GRAPH] = _graph_to_dict(sample_graph)
    mock_tool_context.state[STATE_TURN] = 0
    
    with patch("ts_agent.zones.zone2.tools_ml_auto.SegmentPredictor") as MockPredictor:
        MockPredictor.return_value.predict.side_effect = Exception("ML model error")
        
        result = await record_consumer_answer_ml_auto(
            char_id="CHAR-F2A-I1",
            value="500",
            tool_context=mock_tool_context,
        )
    
    # Should still succeed in recording answer
    assert result["success"] is True
    # But ML prediction should show error
    assert result["ml_prediction"]["confidence"] == 0.0
    assert result["ready_for_match"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
