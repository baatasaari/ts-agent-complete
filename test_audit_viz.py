"""
Quick test of the new Decision Reconstruction System components.
"""

import sys
sys.path.insert(0, "/Users/iArk/Documents/ts-agent-e2e/ts_agent")

from ts_agent.visualiser.app import build_demo_store
from ts_agent.visualiser.components.decision_spine import build_decision_spine
from ts_agent.visualiser.components.decision_justification import build_decision_justification

# Build demo store
print("Building demo store...")
store = build_demo_store()
print(f"Loaded {store.record_count()} sessions")

# Get first EMIT session
emit_session = next((r for r in store.all_records() if r.gate_disposition == "EMIT"), None)
if emit_session:
    print(f"\nTesting with session: {emit_session.session_id[:16]}...")
    print(f"Gate: {emit_session.gate_disposition}")
    print(f"Intent: {emit_session.intent_id}")
    print(f"Matched Segment: {emit_session.matched_segment_id}")
    print(f"Rules: {len(emit_session.rule_evaluations)}")
    
    # Test Decision Spine
    print("\n--- Testing Decision Spine Component ---")
    try:
        spine = build_decision_spine(emit_session, active_zone=4)
        print("✓ Decision Spine built successfully")
        print(f"  Component type: {type(spine)}")
    except Exception as e:
        print(f"✗ Decision Spine failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Decision Justification
    print("\n--- Testing Decision Justification Component ---")
    try:
        justification = build_decision_justification(emit_session)
        print("✓ Decision Justification built successfully")
        print(f"  Component type: {type(justification)}")
    except Exception as e:
        print(f"✗ Decision Justification failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ All components built successfully!")

else:
    print("No EMIT sessions found in store")

# Get a SUPPRESS session
suppress_session = next((r for r in store.all_records() if r.gate_disposition == "SUPPRESS"), None)
if suppress_session:
    print(f"\n\nTesting with SUPPRESS session: {suppress_session.session_id[:16]}...")
    print(f"Gate: {suppress_session.gate_disposition}")
    
    try:
        spine = build_decision_spine(suppress_session, active_zone=3)
        justification = build_decision_justification(suppress_session)
        print("✓ SUPPRESS session components built successfully")
    except Exception as e:
        print(f"✗ SUPPRESS session failed: {e}")
        import traceback
        traceback.print_exc()

print("\n=== Test Complete ===")
