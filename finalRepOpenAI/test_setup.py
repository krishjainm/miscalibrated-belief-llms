#!/usr/bin/env python3
"""
Test script to verify WMAC 2026 research setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test critical imports"""
    try:
        from game_environment.advanced_collusion_agent import AdvancedCollusionAgent
        from game_environment.mixed_player_communication_game import MixedPlayerCommunicationGame
        from team_coordination_engine import TeamCoordinationEngine
        from wmac2026.run_wmac import main
        print("✅ All critical imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    required_dirs = ["game_environment", "wmac2026", "analysis", "texasholdem"]
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Directory structure correct")
    return True

def main():
    """Run all tests"""
    print("🧪 Testing WMAC 2026 research setup...")
    
    tests = [
        test_imports,
        test_directory_structure
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Setup verification successful!")
        print("\n🚀 Ready to run research simulations:")
        print("   python3 wmac2026/run_wmac.py --hands 20 --collusion-llm-players 2 --coordination-mode emergent_only --output-dir data/test_simulation")
    else:
        print("❌ Setup verification failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
