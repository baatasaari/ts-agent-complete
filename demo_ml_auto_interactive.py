"""
Interactive Demo: ML-Automatic Zone 2 Gap-Fill

This demo shows the ML-automatic system in action:
- ML predicts after each answer automatically
- Questions prioritized by SHAP + config
- Shows confidence improving with each answer
- Stops when confidence reaches 75%
"""
import sys
import json
import asyncio
from unittest.mock import Mock

# Add current directory to path
sys.path.insert(0, '.')

from ts_agent.domain.models import TraitGraph, NodeState
from ts_agent.zones.zone2.tools_ml_auto import (
    record_consumer_answer_ml_auto,
    get_next_question_ml_prioritized,
    check_graph_completeness_ml_aware,
    STATE_GRAPH,
    STATE_TURN,
    STATE_PREDICTION_HISTORY,
)
from ts_agent.zones.zone2.tools import _graph_to_dict


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_ml_prediction(ml_data):
    """Display ML prediction results."""
    if not ml_data or not ml_data.get("segment_id"):
        print("   [No prediction yet - need more data]")
        return
    
    print(f"   🤖 ML Prediction:")
    print(f"      Segment: {ml_data['segment_id']}")
    print(f"      Confidence: {ml_data['confidence']:.1%}")
    print(f"      Delta: {ml_data['confidence_delta']:+.1%}")
    
    if ml_data.get('high_confidence'):
        print(f"      ✅ HIGH CONFIDENCE - Ready to stop!")
    else:
        print(f"      ⏳ Keep asking questions...")


def print_top_segments(segments):
    """Display top segment predictions."""
    if not segments:
        return
    print(f"\n   📊 Top 3 Segments:")
    for i, seg in enumerate(segments[:3], 1):
        print(f"      {i}. {seg['label']}: {seg['score']:.1%}")


async def run_interactive_demo():
    """Run the interactive gap-fill demo."""
    print_header("ML-Automatic Gap-Fill Demo")
    print("This demo shows how the system:")
    print("  • Predicts segment after EACH answer (automatically)")
    print("  • Prioritizes questions by SHAP + config")
    print("  • Stops when confidence reaches 75%\n")
    
    input("Press Enter to start...")
    
    # Create mock graph for investment scenario
    graph = TraitGraph(
        session_id="demo-session-001",
        party_ref="DEMO-PARTY",
        intent_id="INT-INVEST",
        situation_id="SIT-INVEST",
    )
    
    # Add investment-related traits
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
    graph.add_node(
        node_id="N4", char_id="CHAR-P2A-I1",
        branch="personal", label="age",
        op=">=", target_value=30, fill_priority=4,
        fill_question_key="age",
    )
    graph.add_node(
        node_id="N5", char_id="CHAR-P2B-I1",
        branch="personal", label="risk_tolerance",
        op="==", target_value="MODERATE", fill_priority=5,
        fill_question_key="risk_tolerance",
    )
    
    # Mock tool context
    mock_context = Mock()
    mock_context.state = {
        STATE_GRAPH: _graph_to_dict(graph),
        STATE_TURN: 0,
        STATE_PREDICTION_HISTORY: "[]",
    }
    
    turn = 0
    
    while True:
        turn += 1
        print_header(f"Turn {turn}")
        
        # Get next question using ML prioritization
        question_result = await get_next_question_ml_prioritized(mock_context)
        
        if question_result.get("done"):
            print("✅ All questions answered!")
            break
        
        char_id = question_result["char_id"]
        label = question_result["label"]
        reason = question_result.get("priority_reason", "Standard priority")
        remaining = question_result.get("remaining", 0)
        ml_conf = question_result.get("current_ml_confidence", 0.0)
        
        print(f"Question Priority: {reason}")
        if ml_conf > 0:
            print(f"Current ML Confidence: {ml_conf:.1%}")
        print(f"Remaining Questions: {remaining}")
        print()
        
        # Ask the question
        print(f"❓ Question: What is your {label}?")
        print(f"   (char_id: {char_id})")
        print()
        
        # Get user input
        answer = input("👤 Your answer: ").strip()
        
        if not answer:
            print("❌ No answer provided. Exiting...")
            break
        
        # Record answer (ML prediction happens automatically!)
        print(f"\n📝 Recording answer...")
        result = await record_consumer_answer_ml_auto(
            char_id=char_id,
            value=answer,
            tool_context=mock_context,
        )
        
        if not result.get("success"):
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            break
        
        print(f"✅ Recorded: {result['node_label']} = {answer}")
        print(f"   Graph Completeness: {result['completeness']:.1%}")
        
        # Display ML prediction (automatic!)
        print_ml_prediction(result.get("ml_prediction"))
        print_top_segments(result.get("top_3_segments", []))
        
        # Check if ready
        if result.get("ready_for_match"):
            print_header("🎯 READY FOR MATCHING!")
            ml_pred = result.get("ml_prediction", {})
            print(f"Final Segment: {ml_pred.get('segment_id')}")
            print(f"Final Confidence: {ml_pred.get('confidence', 0):.1%}")
            print(f"Total Questions Asked: {turn}")
            print()
            
            # Show prediction history
            history_json = mock_context.state.get(STATE_PREDICTION_HISTORY, "[]")
            history = json.loads(history_json)
            
            print("📈 Confidence Evolution:")
            for entry in history:
                print(f"   Turn {entry['turn']}: {entry['top_segment_id']} "
                      f"({entry['top_confidence']:.1%}, Δ{entry['confidence_delta']:+.1%})")
            
            print()
            print("✅ Demo Complete!")
            break
        
        print(f"\n⏭️  Next question: {result.get('next_char_id', 'TBD')}")
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        asyncio.run(run_interactive_demo())
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
