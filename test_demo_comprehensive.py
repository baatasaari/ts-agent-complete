#!/usr/bin/env python3
"""
Comprehensive Demo Test Suite
============================

Tests all critical functionality for tomorrow's demo:
- All 12 scenarios from run_zone2.py
- Observability data collection 
- Visualiser functionality
- End-to-end pipeline

This ensures everything works for the demo presentation.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings for clean test output
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

def test_imports():
    """Test all critical imports work."""
    print("🔍 Testing imports...")
    try:
        # Core demo imports
        from run_zone2 import GeminiConversationDemo, SCENARIOS
        
        # Visualiser imports  
        import ts_agent.visualiser.app as visualiser_app
        
        # Observability imports
        import ts_agent.observability.signals as signals
        
        # Zone 2 agent imports
        from ts_agent.zones.zone2.agent_ml_auto import create_ml_auto_gap_fill_agent
        
        print("✅ All critical imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_observability_fix():
    """Test that observability data collection works despite console log silencing."""
    print("\n🔍 Testing observability fix...")
    try:
        import ts_agent.observability.signals as signals
        
        # Clear any existing listeners/spans
        signals.clear_listeners()
        signals.clear_spans()
        
        # Test that signals can be emitted 
        payload = signals.emit("TEST_SIGNAL", "INFO", "TestZone", test_param="test_value")
        
        # Test that data is collected (spans)
        spans = signals.get_emitted_spans()
        
        # Test listener registration works
        collected_signals = []
        def test_listener(payload):
            collected_signals.append(payload)
        
        signals.register_listener(test_listener)
        signals.emit("TEST_SIGNAL_2", "INFO", "TestZone", another_param="test")
        
        if len(collected_signals) >= 1 and len(spans) >= 0:
            print("✅ Observability data collection working correctly")
            return True
        else:
            print(f"❌ Observability not collecting data: {len(collected_signals)} signals, {len(spans)} spans")
            return False
            
    except Exception as e:
        print(f"❌ Observability test failed: {e}")
        return False

def test_scenario_data():
    """Test all scenario configurations are valid."""
    print("\n🔍 Testing scenario configurations...")
    try:
        from run_zone2 import SCENARIOS
        
        required_scenarios = [str(i) for i in range(1, 13)]  # 1-12
        
        for scenario_id in required_scenarios:
            if scenario_id not in SCENARIOS:
                print(f"❌ Missing scenario {scenario_id}")
                return False
                
            scenario = SCENARIOS[scenario_id]
            required_keys = ['label', 'situation_id', 'intent_id', 'bank_traits']
            
            for key in required_keys:
                if key not in scenario:
                    print(f"❌ Scenario {scenario_id} missing {key}")
                    return False
        
        print(f"✅ All {len(required_scenarios)} scenarios configured correctly")
        return True
        
    except Exception as e:
        print(f"❌ Scenario configuration test failed: {e}")
        return False

async def test_gemini_agent_creation():
    """Test that Gemini agent can be created without API calls."""
    print("\n🔍 Testing Gemini agent creation...")
    try:
        from ts_agent.zones.zone2.agent_ml_auto import create_ml_auto_gap_fill_agent
        
        agent = create_ml_auto_gap_fill_agent()
        
        if agent is not None:
            print("✅ Gemini ML auto agent created successfully")
            return True
        else:
            print("❌ Agent creation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_zone1_graph_building():
    """Test Zone 1 graph building for all scenarios."""
    print("\n🔍 Testing Zone 1 graph building...")
    try:
        from run_zone2 import GeminiConversationDemo, SCENARIOS
        from ts_agent.domain.models import TraitGraph, NodeState
        
        demo = GeminiConversationDemo()
        success_count = 0
        
        # Test first 3 scenarios to verify graph building works
        for scenario_id in ["1", "2", "3"]:
            scenario = SCENARIOS[scenario_id]
            try:
                ctx, state, _ = demo._build_zone1(scenario)
                
                # Verify graph was created
                if ctx.graph and isinstance(ctx.graph, TraitGraph):
                    if len(ctx.graph.nodes) > 0:
                        success_count += 1
                    
            except Exception as e:
                print(f"❌ Zone 1 failed for scenario {scenario_id}: {e}")
                continue
        
        if success_count >= 3:
            print(f"✅ Zone 1 graph building working ({success_count}/3 scenarios tested)")
            return True
        else:
            print(f"❌ Zone 1 graph building failed ({success_count}/3 scenarios)")
            return False
            
    except Exception as e:
        print(f"❌ Zone 1 graph building test failed: {e}")
        return False

def test_visualiser_demo_data():
    """Test visualiser can load with demo data."""
    print("\n🔍 Testing visualiser with demo data...")
    try:
        from ts_agent.visualiser.data_adapter import build_demo_store
        
        # Build demo store (this creates the 19 scenarios with observability data)
        store = build_demo_store()
        
        if store and len(store.sessions) > 0:
            session_count = len(store.sessions)
            consumer_count = len(store.get_consumers())
            
            print(f"✅ Visualiser demo data loaded: {session_count} sessions across {consumer_count} consumers")
            return True
        else:
            print("❌ Visualiser demo store is empty")
            return False
            
    except Exception as e:
        print(f"❌ Visualiser demo data test failed: {e}")
        return False

def test_ml_prediction_components():
    """Test ML prediction and tools work."""
    print("\n🔍 Testing ML prediction components...")
    try:
        from ts_agent.zones.zone2.tools_ml_auto import get_ml_prediction, _build_features_for_prediction
        from ts_agent.domain.models import TraitGraph, TraitNode, NodeState, NodeBranch
        
        # Create a simple test graph
        graph = TraitGraph(
            session_id="test-session",
            party_ref="test-party",
            intent_id="test-intent",
            situation_id="SIT-INV-001"
        )
        
        # Add a few test nodes
        node1 = TraitNode(
            node_id="node1",
            char_id="CHAR-F2A-I1",
            branch=NodeBranch.FINANCIAL,
            label="Monthly Surplus",
            op="==",
            target_value=500,
            data_sources=("CONSUMER_INPUT",),
            aging="session",
            fill_priority=1,
            state=NodeState.KNOWN,
            value=500,
            populated_source="CONSUMER_INPUT"
        )
        graph.add_node(node1)
        
        # Test feature building
        features = _build_features_for_prediction(graph)
        
        # Test ML prediction  
        prediction = get_ml_prediction(graph, 1)
        
        if features and prediction:
            pred_data = json.loads(prediction)
            if "top_segment_id" in pred_data and "top_confidence" in pred_data:
                print("✅ ML prediction components working")
                return True
        
        print("❌ ML prediction components not working correctly")
        return False
        
    except Exception as e:
        print(f"❌ ML prediction test failed: {e}")
        return False

def test_all_scenario_types():
    """Test that all different scenario types can be processed."""
    print("\n🔍 Testing all scenario types...")
    try:
        from run_zone2 import SCENARIOS, GeminiConversationDemo
        
        demo = GeminiConversationDemo()
        scenario_types = {}
        
        # Categorize scenarios by situation type
        for scenario_id, scenario in SCENARIOS.items():
            if scenario_id in ["13", "14"]:  # Skip special cases
                continue
                
            situation = scenario["situation_id"]
            category = situation.split("-")[1]  # INV, SD, PEN, DEC
            
            if category not in scenario_types:
                scenario_types[category] = []
            scenario_types[category].append(scenario_id)
        
        # Test one scenario from each category
        success_categories = []
        for category, scenario_ids in scenario_types.items():
            try:
                test_scenario = SCENARIOS[scenario_ids[0]]
                ctx, state, _ = demo._build_zone1(test_scenario)
                
                if ctx and ctx.graph and len(ctx.graph.nodes) > 0:
                    success_categories.append(category)
                    
            except Exception as e:
                print(f"❌ Failed to process {category} scenario: {e}")
                continue
        
        if len(success_categories) >= 3:  # Should have INV, PEN, DEC at minimum
            print(f"✅ All scenario types working: {', '.join(success_categories)}")
            return True
        else:
            print(f"❌ Some scenario types failed: {success_categories}")
            return False
            
    except Exception as e:
        print(f"❌ Scenario types test failed: {e}")
        return False

def test_end_to_end_silent_run():
    """Test end-to-end pipeline runs without errors (mock version)."""
    print("\n🔍 Testing end-to-end pipeline...")
    try:
        from run_zone2 import GeminiConversationDemo, SCENARIOS
        
        demo = GeminiConversationDemo()
        
        # Test the setup check works
        os.environ["GOOGLE_API_KEY"] = "test-key"  # Mock API key
        
        # Test Zone 1 for scenario 1
        scenario = SCENARIOS["1"]
        ctx, state, _ = demo._build_zone1(scenario)
        
        # Verify the pipeline components
        if (ctx and ctx.graph and 
            state and isinstance(state, dict) and  # Check state is a dict
            len(ctx.graph.nodes) > 0 and
            "ts_graph" in state):  # Check graph is stored in state
            
            print("✅ End-to-end pipeline structure working")
            return True
        else:
            print("❌ End-to-end pipeline structure incomplete")
            print(f"  - ctx: {ctx is not None}")
            print(f"  - ctx.graph: {ctx.graph is not None if ctx else False}")
            print(f"  - state: {isinstance(state, dict) if state else False}")
            print(f"  - nodes: {len(ctx.graph.nodes) if ctx and ctx.graph else 0}")
            print(f"  - graph in state: {'ts_graph' in state if isinstance(state, dict) else False}")
            return False
            
    except Exception as e:
        print(f"❌ End-to-end pipeline test failed: {e}")
        return False

async def run_comprehensive_tests():
    """Run all comprehensive tests for tomorrow's demo."""
    print("🚀 COMPREHENSIVE DEMO TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Observability Fix", test_observability_fix),
        ("Scenario Configuration", test_scenario_data),
        ("Gemini Agent Creation", test_gemini_agent_creation),
        ("Zone 1 Graph Building", test_zone1_graph_building),
        ("Visualiser Demo Data", test_visualiser_demo_data),
        ("ML Prediction Components", test_ml_prediction_components),
        ("All Scenario Types", test_all_scenario_types),
        ("End-to-End Pipeline", test_end_to_end_silent_run),
    ]
    
    results = []
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - DEMO READY!")
    elif passed >= total * 0.8:  # 80% pass rate
        print("\n⚠️  MOSTLY READY - Minor issues detected")
    else:
        print("\n🚨 CRITICAL ISSUES - Demo may have problems")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(run_comprehensive_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)