#!/usr/bin/env python3
"""
Test improved prompts to see if they fix the collusion issues
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from dotenv import load_dotenv
from improved_llm_prompts import get_enhanced_communication_prompt, get_enhanced_collusion_coordination_prompt

# Load environment variables
load_dotenv()

def test_improved_prompts():
    """Test the improved prompts with a sample game state"""
    
    # Sample game state
    game_state = {
        "hole_cards": ["5c", "8c"],
        "board_cards": [],
        "betting_history": "",
        "pot_size": 12,
        "my_chips": 500,
        "my_position": 0,
        "teammate_positions": [1],
        "player_1_chips": 500,
        "player_1_position": 1
    }
    
    teammate_ids = [1]
    chat_history = []
    
    print("🔍 Testing Improved Prompts")
    print("=" * 50)
    
    # Test enhanced communication prompt
    print("\n📝 ENHANCED COMMUNICATION PROMPT:")
    print("-" * 30)
    
    prompt = get_enhanced_communication_prompt(
        game_state=game_state,
        communication_style="emergent",
        teammate_ids=teammate_ids,
        chat_history=chat_history
    )
    
    print(prompt)
    
    print("\n" + "=" * 50)
    print("📊 PROMPT ANALYSIS:")
    print(f"   - Length: {len(prompt)} characters")
    print(f"   - Lines: {len(prompt.split(chr(10)))}")
    print(f"   - Strategic questions: {prompt.count('?')}")
    print(f"   - Learning references: {prompt.count('Remember:')}")
    print(f"   - Strategic context: {'STRATEGIC' in prompt}")
    
    # Test enhanced coordination prompt
    print("\n📝 ENHANCED COORDINATION PROMPT:")
    print("-" * 30)
    
    coord_prompt = get_enhanced_collusion_coordination_prompt(
        game_state=game_state,
        teammate_positions=[1],
        strategy="signal_and_squeeze"
    )
    
    print(coord_prompt)
    
    print("\n" + "=" * 50)
    print("📊 COORDINATION PROMPT ANALYSIS:")
    print(f"   - Length: {len(coord_prompt)} characters")
    print(f"   - Strategic questions: {coord_prompt.count('?')}")
    print(f"   - Learning references: {coord_prompt.count('Remember:')}")
    print(f"   - Coordination opportunities: {coord_prompt.count('SUPPORT')}")

if __name__ == "__main__":
    test_improved_prompts()
