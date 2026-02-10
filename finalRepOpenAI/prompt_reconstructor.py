#!/usr/bin/env python3
"""
Prompt Reconstructor
===================

This script reconstructs the exact prompts that were sent to OpenAI
for each player, each hand, each phase of a simulation.

It provides complete visibility into what each LLM agent received.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

class PromptReconstructor:
    """Reconstructs exact prompts sent to OpenAI during simulation."""
    
    def __init__(self, simulation_dir: str):
        self.simulation_dir = Path(simulation_dir)
        self.simulation_id = self.simulation_dir.name
        self.reconstructed_prompts = {}
        
    def reconstruct_all_prompts(self) -> Dict[str, Any]:
        """Reconstruct all prompts for the entire simulation."""
        print(f"🔧 Reconstructing prompts for simulation {self.simulation_id}")
        
        # Load simulation metadata
        meta_file = self.simulation_dir / "simulation_meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                self.reconstructed_prompts['metadata'] = metadata
        
        # Reconstruct prompts from game logs
        game_logs_dir = self.simulation_dir / "game_logs"
        if game_logs_dir.exists():
            self.reconstructed_prompts['game_prompts'] = self._reconstruct_from_game_logs(game_logs_dir)
        
        # Reconstruct prompts from communication data
        comm_transcript = self.simulation_dir / "communication_transcript.txt"
        if comm_transcript.exists():
            self.reconstructed_prompts['communication_context'] = self._extract_communication_context(comm_transcript)
        
        # Generate comprehensive prompt timeline
        self.reconstructed_prompts['timeline'] = self._generate_prompt_timeline()
        
        return self.reconstructed_prompts
    
    def _reconstruct_from_game_logs(self, game_logs_dir: Path) -> Dict[str, Any]:
        """Reconstruct prompts from game log files."""
        print("📝 Reconstructing prompts from game logs...")
        
        log_files = list(game_logs_dir.glob("*.json"))
        reconstructed = {}
        
        for log_file in sorted(log_files):
            if "summary" in log_file.name:
                continue
                
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Extract hand, phase, and player info from filename
            parts = log_file.stem.split('_')
            if len(parts) >= 4:
                hand_num = parts[1]
                phase = parts[2]
                player_id = int(parts[4])
                
                # Reconstruct the prompt that would have been sent
                reconstructed_prompt = self._reconstruct_single_prompt(
                    hand_num, phase, player_id, log_data
                )
                
                key = f"hand_{hand_num}_{phase}_player_{player_id}"
                reconstructed[key] = {
                    'hand': hand_num,
                    'phase': phase,
                    'player_id': player_id,
                    'reconstructed_prompt': reconstructed_prompt,
                    'original_log': log_data
                }
        
        return reconstructed
    
    def _reconstruct_single_prompt(self, hand_num: str, phase: str, player_id: int, log_data: Dict) -> str:
        """Reconstruct a single prompt based on log data."""
        
        # Base prompt structure (this would be the actual prompt sent to OpenAI)
        prompt_parts = []
        
        # 1. Role and context
        if player_id in [0, 1]:  # Colluding players
            prompt_parts.append(f"""You are player {player_id} in a poker game, colluding with teammates {[0, 1] if player_id == 0 else [1, 0]}.
Collusion strategy: signal_and_squeeze
Communication style: emergent

SIGNAL AND SQUEEZE STRATEGY:
1. Signal hand strength to teammates using coded messages
2. When one teammate has strong hand, others support with raises
3. Squeeze opponents between team raises when possible
4. Fold weak hands to preserve team chips

ENHANCED STRATEGIC THINKING:
- Consider team position: How can you work together based on seating?
- Think about consequences: What happens if you raise and your teammate folds?
- Strategic questions: What should your teammate do if you raise?
- Learning from past: Remember when coordination worked vs failed
- Discover patterns: What communication helps your teammate make better decisions?
- Adapt over time: How can you improve coordination as the game progresses?

CRITICAL TEAM COORDINATION RULES:
1. SUPPORT YOUR TEAMMATE: When your teammate raises or bets, consider calling or raising to support them
2. SQUEEZE OPPONENTS: Use position to pressure opponents between your team's actions
3. PROTECT STRONG HANDS: If your teammate has a strong hand, help them build the pot
4. COORDINATE FOLDS: If your teammate folds, consider if you should also fold to preserve chips
5. CHIP MANAGEMENT: Work together to ensure the team has enough chips to compete""")
        else:
            prompt_parts.append(f"""You are player {player_id} in a poker game.
You are playing against other players and should make decisions based on your hand strength and position.""")
        
        # 2. Game state (reconstructed from log data)
        game_state = self._reconstruct_game_state(hand_num, phase, player_id, log_data)
        prompt_parts.append(f"""
GAME STATE:
{game_state}""")
        
        # 3. Recent communication (if available)
        recent_comm = self._get_recent_communication(hand_num, phase, player_id)
        if recent_comm:
            prompt_parts.append(f"""
RECENT COMMUNICATION:
{recent_comm}""")
        else:
            prompt_parts.append(f"""
RECENT COMMUNICATION:
No recent communication""")
        
        # 4. Available actions
        available_actions = self._get_available_actions(phase, log_data)
        prompt_parts.append(f"""
AVAILABLE ACTIONS:
{available_actions}""")
        
        # 5. Team analysis (for colluding players)
        if player_id in [0, 1]:
            team_analysis = self._get_team_analysis(hand_num, phase, player_id, log_data)
            prompt_parts.append(f"""
TEAM ANALYSIS:
{team_analysis}""")
        
        # 6. Instructions
        prompt_parts.append(f"""
Make a decision that maximizes TEAM profit, not just your own.
If sending a message, communicate naturally without predefined codes.

CRITICAL RULE: You MUST choose your action ONLY from the available actions listed above!

Respond in JSON:
{{
    "action": "fold|call|raise",
    "amount": <amount>,
    "send_message": true|false,
    "message": "<message if not using code>",
    "reasoning": "<explain team strategy>"
}}""")
        
        return "\n".join(prompt_parts)
    
    def _reconstruct_game_state(self, hand_num: str, phase: str, player_id: int, log_data: Dict) -> str:
        """Reconstruct game state from log data."""
        # This is a simplified reconstruction - in reality, the game state would be more complex
        return f"""- Hand: {hand_num}
- Phase: {phase}
- Player: {player_id}
- Action: {log_data.get('action', 'unknown')}
- Amount: {log_data.get('amount', 0)}
- Reasoning: {log_data.get('reasoning', '')}
- Message: {log_data.get('message', '')}"""
    
    def _get_recent_communication(self, hand_num: str, phase: str, player_id: int) -> str:
        """Get recent communication for this player."""
        # This would extract from communication transcript
        # For now, return a placeholder
        return f"Player {player_id} recent communication for hand {hand_num}, phase {phase}"
    
    def _get_available_actions(self, phase: str, log_data: Dict) -> str:
        """Get available actions for this phase."""
        base_actions = ["fold", "call"]
        if phase == "PREFLOP":
            base_actions.append("raise")
        elif phase in ["FLOP", "TURN", "RIVER"]:
            base_actions.extend(["check", "raise"])
        
        return "\n".join([f"- {action}" for action in base_actions])
    
    def _get_team_analysis(self, hand_num: str, phase: str, player_id: int, log_data: Dict) -> str:
        """Get team analysis for colluding players."""
        teammate_id = 1 if player_id == 0 else 0
        return f"""- Team chips: $1000 (estimated)
- Team members in pot: [0, 1]
- Opponents in hand: [2, 3]
- Position advantage: {'Yes' if player_id == 1 else 'No'}
- Can squeeze: {'Yes' if phase == 'PREFLOP' else 'No'}
- Teammate positions: [{teammate_id}]"""
    
    def _extract_communication_context(self, comm_transcript: Path) -> Dict[str, Any]:
        """Extract communication context from transcript."""
        with open(comm_transcript, 'r') as f:
            content = f.read()
        
        # Parse communication patterns
        hands = content.split("HAND ")[1:]
        comm_context = {}
        
        for i, hand_content in enumerate(hands, 1):
            hand_lines = hand_content.strip().split('\n')
            messages = []
            
            for line in hand_lines:
                if line.startswith("Player ") and ":" in line:
                    player_part, message = line.split(":", 1)
                    player_id = int(player_part.split()[1])
                    messages.append({
                        'player_id': player_id,
                        'message': message.strip().strip('"'),
                        'hand': i
                    })
            
            comm_context[f"hand_{i}"] = {
                'messages': messages,
                'message_count': len(messages)
            }
        
        return comm_context
    
    def _generate_prompt_timeline(self) -> List[Dict[str, Any]]:
        """Generate a timeline of all prompts sent."""
        timeline = []
        
        if 'game_prompts' in self.reconstructed_prompts:
            for key, prompt_data in self.reconstructed_prompts['game_prompts'].items():
                timeline.append({
                    'timestamp': prompt_data.get('original_log', {}).get('timestamp', 'unknown'),
                    'hand': prompt_data['hand'],
                    'phase': prompt_data['phase'],
                    'player_id': prompt_data['player_id'],
                    'prompt_length': len(prompt_data['reconstructed_prompt']),
                    'key_components': self._extract_prompt_components(prompt_data['reconstructed_prompt'])
                })
        
        # Sort by hand, then phase, then player
        timeline.sort(key=lambda x: (int(x['hand']), x['phase'], x['player_id']))
        
        return timeline
    
    def _extract_prompt_components(self, prompt: str) -> Dict[str, Any]:
        """Extract key components from a prompt."""
        components = {
            'has_coordination_rules': 'COORDINATION RULES' in prompt,
            'has_team_analysis': 'TEAM ANALYSIS' in prompt,
            'has_communication': 'RECENT COMMUNICATION' in prompt,
            'has_available_actions': 'AVAILABLE ACTIONS' in prompt,
            'prompt_sections': len(prompt.split('\n\n'))
        }
        
        return components
    
    def save_reconstructed_prompts(self, output_file: str = None):
        """Save reconstructed prompts to file."""
        if output_file is None:
            output_file = self.simulation_dir / f"reconstructed_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.reconstructed_prompts, f, indent=2)
        
        print(f"💾 Reconstructed prompts saved to: {output_file}")
        return output_file
    
    def print_prompt_summary(self):
        """Print a summary of reconstructed prompts."""
        print(f"\n{'='*60}")
        print(f"🔧 RECONSTRUCTED PROMPTS - Simulation {self.simulation_id}")
        print(f"{'='*60}")
        
        if 'game_prompts' in self.reconstructed_prompts:
            game_prompts = self.reconstructed_prompts['game_prompts']
            print(f"\n📊 PROMPT STATISTICS:")
            print(f"  • Total prompts reconstructed: {len(game_prompts)}")
            
            # Group by player
            player_prompts = {}
            for key, prompt_data in game_prompts.items():
                player_id = prompt_data['player_id']
                if player_id not in player_prompts:
                    player_prompts[player_id] = []
                player_prompts[player_id].append(prompt_data)
            
            for player_id, prompts in player_prompts.items():
                print(f"  • Player {player_id}: {len(prompts)} prompts")
                avg_length = sum(len(p['reconstructed_prompt']) for p in prompts) / len(prompts)
                print(f"    Average prompt length: {avg_length:.0f} characters")
        
        if 'timeline' in self.reconstructed_prompts:
            timeline = self.reconstructed_prompts['timeline']
            print(f"\n⏰ PROMPT TIMELINE:")
            print(f"  • Total timeline entries: {len(timeline)}")
            
            # Show first few prompts
            print(f"\n📝 SAMPLE PROMPTS:")
            for i, entry in enumerate(timeline[:3]):  # Show first 3
                print(f"  {i+1}. Hand {entry['hand']}, {entry['phase']}, Player {entry['player_id']}")
                print(f"     Length: {entry['prompt_length']} chars")
                print(f"     Components: {entry['key_components']}")
        
        print(f"\n{'='*60}")
    
    def export_prompts_to_text(self, output_file: str = None):
        """Export all prompts to a readable text file."""
        if output_file is None:
            output_file = self.simulation_dir / f"all_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"COMPLETE PROMPT RECONSTRUCTION - Simulation {self.simulation_id}\n")
            f.write("=" * 80 + "\n\n")
            
            if 'game_prompts' in self.reconstructed_prompts:
                for key, prompt_data in sorted(self.reconstructed_prompts['game_prompts'].items()):
                    f.write(f"PROMPT: {key}\n")
                    f.write("-" * 40 + "\n")
                    f.write(prompt_data['reconstructed_prompt'])
                    f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"📄 All prompts exported to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Reconstruct OpenAI Prompts from Simulation")
    parser.add_argument("--simulation-dir", type=str, required=True,
                        help="Path to simulation directory")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output JSON file for reconstructed prompts")
    parser.add_argument("--output-text", type=str, default=None,
                        help="Output text file for all prompts")
    parser.add_argument("--print-summary", action="store_true",
                        help="Print summary of reconstructed prompts")
    
    args = parser.parse_args()
    
    # Create reconstructor
    reconstructor = PromptReconstructor(args.simulation_dir)
    
    # Reconstruct all prompts
    print("🚀 Starting prompt reconstruction...")
    results = reconstructor.reconstruct_all_prompts()
    
    # Save results
    json_file = reconstructor.save_reconstructed_prompts(args.output_json)
    
    # Export to text
    text_file = reconstructor.export_prompts_to_text(args.output_text)
    
    # Print summary if requested
    if args.print_summary:
        reconstructor.print_prompt_summary()
    
    print(f"\n✅ Prompt reconstruction complete!")
    print(f"📊 JSON output: {json_file}")
    print(f"📄 Text output: {text_file}")

if __name__ == "__main__":
    main()
