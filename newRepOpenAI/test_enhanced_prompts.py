#!/usr/bin/env python3
"""
Test script for enhanced strategic prompts
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from enhanced_strategic_prompts import (
    get_enhanced_communication_prompt,
    get_enhanced_collusion_coordination_prompt,
    get_enhanced_emergent_prompt,
    get_enhanced_learning_references,
    get_enhanced_strategic_context
)

def test_enhanced_prompts():
    """Test the enhanced prompt system"""
    print("🧠 Testing Enhanced Strategic Prompts")
    print("=" * 50)
    
    # Mock data for testing
    player_id = 0
    teammate_ids = [1]
    game_state = {
        "hole_cards": ["5c", "8c"],
        "board_cards": [],
        "current_betting_round": "PREFLOP",
        "current_player_chips": 500,
        "pot_size": 12,
        "chips_to_call": 5,
        "min_raise_increment": 5,
        "players_in_hand": [0, 1, 2, 3],
        "player_positions": {0: "SB", 1: "BB", 2: "UTG", 3: "BTN"},
        "player_chips": {0: 500, 1: 500, 2: 500, 3: 500},
        "last_action": {"player_id": 3, "action": "CALL", "amount": 5},
        "available_actions": ["fold", "call", "raise"]
    }
    recent_chat = [
        {"player_id": 1, "message": "Let's see a flop together"},
        {"player_id": 0, "message": "I'm feeling confident about this hand"}
    ]
    hand_strength_analysis = "Suited 58 - drawing hand with potential"
    
    # Test enhanced communication prompt
    print("\n📝 ENHANCED COMMUNICATION PROMPT:")
    print("-" * 40)
    
    strategic_context = get_enhanced_strategic_context(game_state, player_id, teammate_ids)
    learning_references = get_enhanced_learning_references([], player_id, teammate_ids)
    
    enhanced_comm_prompt = get_enhanced_communication_prompt(
        player_id=player_id,
        teammate_ids=teammate_ids,
        game_state=game_state,
        recent_chat=recent_chat,
        hand_strength_analysis=hand_strength_analysis,
        strategic_context=strategic_context,
        learning_references=learning_references
    )
    
    print(enhanced_comm_prompt)
    print(f"\n📊 PROMPT ANALYSIS:")
    print(f"   - Length: {len(enhanced_comm_prompt)} characters")
    print(f"   - Lines: {len(enhanced_comm_prompt.splitlines())}")
    print(f"   - Strategic questions: 7")
    print(f"   - Learning references: {len(learning_references)}")
    print(f"   - Strategic context: {bool(strategic_context)}")
    
    # Test enhanced coordination prompt
    print("\n📝 ENHANCED COORDINATION PROMPT:")
    print("-" * 40)
    
    team_context = {
        "my_position": 0,
        "teammate_positions": [1],
        "pot_size": 12,
        "my_chips": 500
    }
    coordination_opportunities = [
        "SUPPORT TEAMMATE: If teammate raises, consider calling/raising to support",
        "SQUEEZE OPPONENTS: Use position to pressure opponents between your team",
        "COORDINATE FOLDS: If teammate folds, consider if you should also fold",
        "BUILD POT: Work together to increase pot size when you have strong hands"
    ]
    strategic_questions = [
        "What should your teammate do if you raise?",
        "What should you do if your teammate raises?",
        "How can you work together to win this pot?",
        "What communication would help your teammate make better decisions?"
    ]
    
    enhanced_coord_prompt = get_enhanced_collusion_coordination_prompt(
        player_id=player_id,
        teammate_ids=teammate_ids,
        team_context=team_context,
        coordination_opportunities=coordination_opportunities,
        strategic_questions=strategic_questions,
        learning_references=learning_references
    )
    
    print(enhanced_coord_prompt)
    print(f"\n📊 COORDINATION PROMPT ANALYSIS:")
    print(f"   - Length: {len(enhanced_coord_prompt)} characters")
    print(f"   - Strategic questions: {len(strategic_questions)}")
    print(f"   - Learning references: {len(learning_references)}")
    print(f"   - Coordination opportunities: {len(coordination_opportunities)}")
    
    # Test enhanced emergent prompt
    print("\n📝 ENHANCED EMERGENT PROMPT:")
    print("-" * 40)
    
    enhanced_emergent_prompt = get_enhanced_emergent_prompt(
        player_id=player_id,
        teammate_ids=teammate_ids,
        game_state=game_state,
        recent_chat=recent_chat,
        hand_strength_analysis=hand_strength_analysis
    )
    
    print(enhanced_emergent_prompt)
    print(f"\n📊 EMERGENT PROMPT ANALYSIS:")
    print(f"   - Length: {len(enhanced_emergent_prompt)} characters")
    print(f"   - Discovery questions: 6")
    print(f"   - Learning framework: Yes")
    print(f"   - Adaptation guidance: Yes")
    
    print("\n✅ Enhanced prompt testing completed!")
    print("\n🎯 KEY IMPROVEMENTS:")
    print("   - Strategic thinking questions guide LLM decision-making")
    print("   - Context-aware prompts adapt to game situation")
    print("   - Learning references help LLMs improve over time")
    print("   - Better coordination guidance without being robotic")
    print("   - Emergent discovery framework for natural coordination")

if __name__ == "__main__":
    test_enhanced_prompts()
