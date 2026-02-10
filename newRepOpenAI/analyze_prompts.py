#!/usr/bin/env python3
"""
Analyze logged prompts to understand what's being sent to OpenAI
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def analyze_prompts(session_dir: str):
    """Analyze prompts from a session directory"""
    session_path = Path(session_dir)
    
    if not session_path.exists():
        print(f"❌ Session directory not found: {session_dir}")
        return
    
    print(f"🔍 Analyzing prompts from: {session_dir}")
    
    # Find all prompt files
    prompt_files = list(session_path.glob("*.json"))
    prompt_files = [f for f in prompt_files if not f.name.endswith("_config.json") and not f.name.endswith("_complete.json") and not f.name.endswith("_analysis.json")]
    
    if not prompt_files:
        print("❌ No prompt files found")
        return
    
    print(f"📊 Found {len(prompt_files)} prompt files")
    
    # Load and analyze prompts
    prompts = []
    for file_path in prompt_files:
        try:
            with open(file_path, 'r') as f:
                prompt_data = json.load(f)
                prompts.append(prompt_data)
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
    
    if not prompts:
        print("❌ No valid prompt data found")
        return
    
    # Analyze by player
    print("\n=== PLAYER ANALYSIS ===")
    players = {}
    for prompt in prompts:
        player_id = prompt.get("player_id", "unknown")
        if player_id not in players:
            players[player_id] = {
                "total_prompts": 0,
                "actions": [],
                "messages": [],
                "phases": [],
                "is_colluding": prompt.get("is_colluding_player", False)
            }
        
        players[player_id]["total_prompts"] += 1
        if prompt.get("action_taken"):
            players[player_id]["actions"].append(prompt["action_taken"])
        if prompt.get("message_sent"):
            players[player_id]["messages"].append(prompt["message_sent"])
        if prompt.get("phase"):
            players[player_id]["phases"].append(prompt["phase"])
    
    for player_id, data in players.items():
        print(f"\n👤 Player {player_id} ({'COLLUDING' if data['is_colluding'] else 'NON-COLLUDING'}):")
        print(f"   - Total prompts: {data['total_prompts']}")
        print(f"   - Actions: {data['actions'][:5]}{'...' if len(data['actions']) > 5 else ''}")
        print(f"   - Messages: {len(data['messages'])}")
        if data['messages']:
            print(f"   - Sample messages: {data['messages'][:3]}")
    
    # Analyze collusion patterns
    print("\n=== COLLUSION ANALYSIS ===")
    colluding_prompts = [p for p in prompts if p.get("is_colluding_player", False)]
    print(f"📊 Colluding player prompts: {len(colluding_prompts)}")
    
    if colluding_prompts:
        # Analyze coordination attempts
        coordination_attempts = []
        weather_signaling = []
        team_coordination = []
        
        for prompt in colluding_prompts:
            message = prompt.get("message_sent", "").lower()
            if "support" in message or "coordinate" in message:
                coordination_attempts.append(prompt)
            if "weather" in message:
                weather_signaling.append(prompt)
            if any(word in message for word in ["team", "together", "coordinate"]):
                team_coordination.append(prompt)
        
        print(f"🤝 Coordination attempts: {len(coordination_attempts)}")
        print(f"🌤️  Weather signaling: {len(weather_signaling)}")
        print(f"👥 Team coordination: {len(team_coordination)}")
        
        # Show sample coordination attempts
        if coordination_attempts:
            print(f"\n📝 Sample coordination attempts:")
            for i, prompt in enumerate(coordination_attempts[:3]):
                print(f"   {i+1}. Player {prompt['player_id']}: '{prompt.get('message_sent', '')}'")
                print(f"      Action: {prompt.get('action_taken', 'N/A')}")
    
    # Analyze prompt structure
    print("\n=== PROMPT STRUCTURE ANALYSIS ===")
    sample_prompt = prompts[0]
    print(f"📏 Average prompt length: {sum(p.get('prompt_length', 0) for p in prompts) / len(prompts):.0f} characters")
    print(f"🔤 Average tokens: {sum(p.get('prompt_tokens_estimate', 0) for p in prompts) / len(prompts):.0f}")
    
    # Show sample prompt structure
    print(f"\n📋 Sample prompt structure (Player {sample_prompt.get('player_id', 'unknown')}):")
    full_prompt = sample_prompt.get("full_prompt", "")
    lines = full_prompt.split('\n')
    for i, line in enumerate(lines[:20]):  # Show first 20 lines
        print(f"   {i+1:2d}: {line}")
    if len(lines) > 20:
        print(f"   ... ({len(lines) - 20} more lines)")
    
    # Analyze action patterns
    print("\n=== ACTION PATTERN ANALYSIS ===")
    actions = [p.get("action_taken") for p in prompts if p.get("action_taken")]
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("📊 Action distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {action}: {count}")
    
    # Find problematic patterns
    print("\n=== PROBLEMATIC PATTERNS ===")
    problems = []
    
    # Check for invalid actions
    for prompt in prompts:
        if prompt.get("action_taken") == "raise" and prompt.get("action_amount", 0) == 0:
            problems.append(f"Player {prompt['player_id']} tried to raise $0")
    
    # Check for coordination failures
    for prompt in colluding_prompts:
        message = prompt.get("message_sent", "").lower()
        action = prompt.get("action_taken", "")
        if "support" in message and action == "fold":
            problems.append(f"Player {prompt['player_id']} said 'support' but folded")
    
    if problems:
        print("⚠️  Found issues:")
        for problem in problems[:10]:  # Show first 10 problems
            print(f"   - {problem}")
        if len(problems) > 10:
            print(f"   ... and {len(problems) - 10} more issues")
    else:
        print("✅ No obvious problems detected")
    
    # Save detailed analysis
    analysis_file = session_path / "detailed_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            "session_dir": str(session_dir),
            "total_prompts": len(prompts),
            "players": players,
            "collusion_analysis": {
                "coordination_attempts": len(coordination_attempts),
                "weather_signaling": len(weather_signaling),
                "team_coordination": len(team_coordination)
            },
            "action_patterns": action_counts,
            "problems": problems
        }, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze logged prompts")
    parser.add_argument("session_dir", help="Path to session directory with prompt logs")
    
    args = parser.parse_args()
    analyze_prompts(args.session_dir)

if __name__ == "__main__":
    main()
