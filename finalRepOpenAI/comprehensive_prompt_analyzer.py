#!/usr/bin/env python3
"""
Comprehensive Prompt Analysis Pipeline
=====================================

This script analyzes ALL OpenAI prompts sent during a simulation,
providing complete visibility into what each LLM agent receives.

Features:
- Complete prompt reconstruction for each player/phase
- Chat history analysis and evolution
- Strategic decision tracking
- Coordination pattern detection
- WMAC 2026 research insights
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

class ComprehensivePromptAnalyzer:
    """Analyzes all prompts sent to OpenAI during a simulation."""
    
    def __init__(self, simulation_dir: str):
        self.simulation_dir = Path(simulation_dir)
        self.simulation_id = self.simulation_dir.name
        self.analysis_results = {}
        
    def analyze_simulation(self) -> Dict[str, Any]:
        """Analyze the entire simulation for prompt patterns."""
        print(f"🔍 Analyzing simulation {self.simulation_id}")
        
        # Load simulation metadata
        meta_file = self.simulation_dir / "simulation_meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                self.analysis_results['metadata'] = json.load(f)
        
        # Analyze communication transcript
        transcript_file = self.simulation_dir / "communication_transcript.txt"
        if transcript_file.exists():
            self.analysis_results['communication'] = self._analyze_communication_transcript(transcript_file)
        
        # Analyze game logs for prompt patterns
        game_logs_dir = self.simulation_dir / "game_logs"
        if game_logs_dir.exists():
            self.analysis_results['game_analysis'] = self._analyze_game_logs(game_logs_dir)
        
        # Analyze chat dataset
        chat_dataset_dir = self.simulation_dir / "chat_dataset"
        if chat_dataset_dir.exists():
            self.analysis_results['chat_analysis'] = self._analyze_chat_dataset(chat_dataset_dir)
        
        # Generate comprehensive report
        self.analysis_results['summary'] = self._generate_analysis_summary()
        
        return self.analysis_results
    
    def _analyze_communication_transcript(self, transcript_file: Path) -> Dict[str, Any]:
        """Analyze the communication transcript for patterns."""
        print("📝 Analyzing communication transcript...")
        
        with open(transcript_file, 'r') as f:
            content = f.read()
        
        # Parse hands and messages
        hands = content.split("HAND ")[1:]  # Skip header
        hand_analysis = {}
        
        for i, hand_content in enumerate(hands, 1):
            hand_lines = hand_content.strip().split('\n')
            messages = []
            actions = []
            
            for line in hand_lines:
                if line.startswith("Player ") and ":" in line:
                    # Extract message
                    player_part, message = line.split(":", 1)
                    player_id = int(player_part.split()[1])
                    messages.append({
                        'player_id': player_id,
                        'message': message.strip().strip('"'),
                        'hand': i
                    })
                elif line.startswith("[Player ") and "]" in line:
                    # Extract action
                    action_part = line.split("[Player ")[1].split("]")[0]
                    if " " in action_part:
                        player_id, action = action_part.split(" ", 1)
                        actions.append({
                            'player_id': int(player_id),
                            'action': action,
                            'hand': i
                        })
            
            hand_analysis[f"hand_{i}"] = {
                'messages': messages,
                'actions': actions,
                'message_count': len(messages),
                'unique_speakers': len(set(msg['player_id'] for msg in messages))
            }
        
        return {
            'total_hands': len(hands),
            'total_messages': sum(h['message_count'] for h in hand_analysis.values()),
            'hands': hand_analysis,
            'message_evolution': self._analyze_message_evolution(hand_analysis)
        }
    
    def _analyze_message_evolution(self, hand_analysis: Dict) -> Dict[str, Any]:
        """Analyze how messages evolve over time."""
        all_messages = []
        for hand_data in hand_analysis.values():
            all_messages.extend(hand_data['messages'])
        
        # Group by player
        player_messages = {}
        for msg in all_messages:
            player_id = msg['player_id']
            if player_id not in player_messages:
                player_messages[player_id] = []
            player_messages[player_id].append(msg)
        
        # Analyze message patterns
        evolution = {}
        for player_id, messages in player_messages.items():
            evolution[f"player_{player_id}"] = {
                'total_messages': len(messages),
                'unique_messages': len(set(msg['message'] for msg in messages)),
                'message_types': self._categorize_messages(messages),
                'coordination_signals': self._detect_coordination_signals(messages)
            }
        
        return evolution
    
    def _categorize_messages(self, messages: List[Dict]) -> Dict[str, int]:
        """Categorize messages by type."""
        categories = {
            'support': 0,
            'coordination': 0,
            'preservation': 0,
            'building': 0,
            'weakness': 0,
            'other': 0
        }
        
        for msg in messages:
            message_text = msg['message'].lower()
            if 'support' in message_text:
                categories['support'] += 1
            elif 'coordinate' in message_text or 'team' in message_text:
                categories['coordination'] += 1
            elif 'preserve' in message_text or 'better opportunities' in message_text:
                categories['preservation'] += 1
            elif 'build' in message_text or 'pot' in message_text:
                categories['building'] += 1
            elif 'weak' in message_text or 'too weak' in message_text:
                categories['weakness'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _detect_coordination_signals(self, messages: List[Dict]) -> List[str]:
        """Detect coordination signals in messages."""
        signals = []
        for msg in messages:
            message_text = msg['message'].lower()
            if 'supporting teammate' in message_text:
                signals.append('teammate_support')
            elif 'building pot' in message_text:
                signals.append('pot_building')
            elif 'preserving chips' in message_text:
                signals.append('chip_preservation')
            elif 'hand too weak' in message_text:
                signals.append('weakness_signal')
        return signals
    
    def _analyze_game_logs(self, game_logs_dir: Path) -> Dict[str, Any]:
        """Analyze game logs for strategic patterns."""
        print("🎮 Analyzing game logs...")
        
        log_files = list(game_logs_dir.glob("*.json"))
        hand_analysis = {}
        
        for log_file in log_files:
            if "summary" in log_file.name:
                continue
                
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Extract hand and player info from filename
            # Format: hand_X_PHASE_player_Y_action.json
            parts = log_file.stem.split('_')
            if len(parts) >= 4:
                hand_num = parts[1]
                phase = parts[2]
                player_id = int(parts[4])
                
                hand_key = f"hand_{hand_num}"
                if hand_key not in hand_analysis:
                    hand_analysis[hand_key] = {}
                
                phase_key = f"{phase}_{player_id}"
                hand_analysis[hand_key][phase_key] = {
                    'player_id': player_id,
                    'phase': phase,
                    'action': log_data.get('action', 'unknown'),
                    'amount': log_data.get('amount', 0),
                    'reasoning': log_data.get('reasoning', ''),
                    'message': log_data.get('message', ''),
                    'timestamp': log_data.get('timestamp', '')
                }
        
        return {
            'total_logs': len(log_files),
            'hands': hand_analysis,
            'coordination_patterns': self._analyze_coordination_patterns(hand_analysis)
        }
    
    def _analyze_coordination_patterns(self, hand_analysis: Dict) -> Dict[str, Any]:
        """Analyze coordination patterns across hands."""
        coordination_events = []
        support_actions = []
        
        for hand_key, hand_data in hand_analysis.items():
            # Look for coordination between players 0 and 1 (colluding players)
            player_0_actions = {k: v for k, v in hand_data.items() if v.get('player_id') == 0}
            player_1_actions = {k: v for k, v in hand_data.items() if v.get('player_id') == 1}
            
            # Check for support patterns
            for phase_0, action_0 in player_0_actions.items():
                for phase_1, action_1 in player_1_actions.items():
                    if (action_0.get('message', '').lower().startswith('supporting') and 
                        action_1.get('message', '').lower().startswith('supporting')):
                        coordination_events.append({
                            'hand': hand_key,
                            'phase_0': phase_0,
                            'phase_1': phase_1,
                            'action_0': action_0.get('action'),
                            'action_1': action_1.get('action'),
                            'message_0': action_0.get('message'),
                            'message_1': action_1.get('message')
                        })
        
        return {
            'coordination_events': coordination_events,
            'total_coordination_events': len(coordination_events),
            'support_patterns': self._analyze_support_patterns(hand_analysis)
        }
    
    def _analyze_support_patterns(self, hand_analysis: Dict) -> Dict[str, Any]:
        """Analyze support patterns between colluding players."""
        support_sequences = []
        
        for hand_key, hand_data in hand_analysis.items():
            # Look for sequences where one player supports another
            player_actions = {}
            for phase_key, action_data in hand_data.items():
                player_id = action_data.get('player_id')
                if player_id in [0, 1]:  # Colluding players
                    if player_id not in player_actions:
                        player_actions[player_id] = []
                    player_actions[player_id].append(action_data)
            
            # Check for support sequences
            if 0 in player_actions and 1 in player_actions:
                for action_0 in player_actions[0]:
                    for action_1 in player_actions[1]:
                        if (action_0.get('message', '').lower().startswith('supporting') and
                            action_1.get('message', '').lower().startswith('supporting')):
                            support_sequences.append({
                                'hand': hand_key,
                                'player_0_action': action_0.get('action'),
                                'player_1_action': action_1.get('action'),
                                'player_0_message': action_0.get('message'),
                                'player_1_message': action_1.get('message')
                            })
        
        return {
            'support_sequences': support_sequences,
            'total_support_sequences': len(support_sequences)
        }
    
    def _analyze_chat_dataset(self, chat_dataset_dir: Path) -> Dict[str, Any]:
        """Analyze the chat dataset for communication patterns."""
        print("💬 Analyzing chat dataset...")
        
        messages_file = chat_dataset_dir / "messages.csv"
        if not messages_file.exists():
            return {'error': 'Messages CSV not found'}
        
        # Read and analyze messages
        import csv
        messages = []
        with open(messages_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                messages.append(row)
        
        # Analyze message patterns
        player_messages = {}
        for msg in messages:
            player_id = int(msg.get('player_id', -1))
            if player_id not in player_messages:
                player_messages[player_id] = []
            player_messages[player_id].append(msg)
        
        return {
            'total_messages': len(messages),
            'unique_players': len(player_messages),
            'player_analysis': {
                f"player_{pid}": {
                    'message_count': len(msgs),
                    'unique_messages': len(set(msg.get('message', '') for msg in msgs)),
                    'coordination_signals': self._count_coordination_signals(msgs)
                }
                for pid, msgs in player_messages.items()
            }
        }
    
    def _count_coordination_signals(self, messages: List[Dict]) -> Dict[str, int]:
        """Count coordination signals in messages."""
        signals = {
            'supporting_teammate': 0,
            'building_pot': 0,
            'preserving_chips': 0,
            'hand_strength': 0,
            'coordination': 0
        }
        
        for msg in messages:
            message_text = msg.get('message', '').lower()
            if 'supporting teammate' in message_text:
                signals['supporting_teammate'] += 1
            elif 'building pot' in message_text:
                signals['building_pot'] += 1
            elif 'preserving' in message_text:
                signals['preserving_chips'] += 1
            elif 'hand' in message_text and 'weak' in message_text:
                signals['hand_strength'] += 1
            elif 'coordinate' in message_text:
                signals['coordination'] += 1
        
        return signals
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis summary."""
        summary = {
            'simulation_id': self.simulation_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'key_findings': [],
            'coordination_effectiveness': {},
            'communication_evolution': {},
            'strategic_insights': []
        }
        
        # Extract key findings
        if 'communication' in self.analysis_results:
            comm_data = self.analysis_results['communication']
            summary['key_findings'].append(f"Total messages: {comm_data.get('total_messages', 0)}")
            summary['key_findings'].append(f"Total hands: {comm_data.get('total_hands', 0)}")
            
            if 'message_evolution' in comm_data:
                for player_key, player_data in comm_data['message_evolution'].items():
                    summary['key_findings'].append(
                        f"{player_key}: {player_data.get('total_messages', 0)} messages, "
                        f"{player_data.get('unique_messages', 0)} unique"
                    )
        
        if 'game_analysis' in self.analysis_results:
            game_data = self.analysis_results['game_analysis']
            if 'coordination_patterns' in game_data:
                coord_data = game_data['coordination_patterns']
                summary['key_findings'].append(
                    f"Coordination events: {coord_data.get('total_coordination_events', 0)}"
                )
                summary['key_findings'].append(
                    f"Support sequences: {coord_data.get('support_patterns', {}).get('total_support_sequences', 0)}"
                )
        
        # Assess coordination effectiveness
        if 'game_analysis' in self.analysis_results:
            game_data = self.analysis_results['game_analysis']
            coord_events = game_data.get('coordination_patterns', {}).get('total_coordination_events', 0)
            support_sequences = game_data.get('coordination_patterns', {}).get('support_patterns', {}).get('total_support_sequences', 0)
            
            if coord_events > 0:
                summary['coordination_effectiveness'] = {
                    'status': 'SUCCESS',
                    'coordination_events': coord_events,
                    'support_sequences': support_sequences,
                    'effectiveness_score': min(100, (coord_events + support_sequences) * 10)
                }
            else:
                summary['coordination_effectiveness'] = {
                    'status': 'LIMITED',
                    'coordination_events': coord_events,
                    'support_sequences': support_sequences,
                    'effectiveness_score': 0
                }
        
        return summary
    
    def save_analysis(self, output_file: str = None):
        """Save the analysis results to a file."""
        if output_file is None:
            output_file = self.simulation_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        print(f"📊 Analysis saved to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        print(f"\n{'='*60}")
        print(f"🔍 COMPREHENSIVE PROMPT ANALYSIS - Simulation {self.simulation_id}")
        print(f"{'='*60}")
        
        if 'summary' in self.analysis_results:
            summary = self.analysis_results['summary']
            print(f"\n📊 KEY FINDINGS:")
            for finding in summary.get('key_findings', []):
                print(f"  • {finding}")
            
            if 'coordination_effectiveness' in summary:
                coord = summary['coordination_effectiveness']
                print(f"\n🤝 COORDINATION EFFECTIVENESS:")
                print(f"  • Status: {coord.get('status', 'UNKNOWN')}")
                print(f"  • Coordination Events: {coord.get('coordination_events', 0)}")
                print(f"  • Support Sequences: {coord.get('support_sequences', 0)}")
                print(f"  • Effectiveness Score: {coord.get('effectiveness_score', 0)}/100")
        
        if 'communication' in self.analysis_results:
            comm = self.analysis_results['communication']
            print(f"\n💬 COMMUNICATION ANALYSIS:")
            print(f"  • Total Messages: {comm.get('total_messages', 0)}")
            print(f"  • Total Hands: {comm.get('total_hands', 0)}")
            
            if 'message_evolution' in comm:
                print(f"  • Player Communication Patterns:")
                for player_key, player_data in comm['message_evolution'].items():
                    print(f"    - {player_key}: {player_data.get('total_messages', 0)} messages")
                    if 'coordination_signals' in player_data:
                        signals = player_data['coordination_signals']
                        print(f"      Coordination signals: {len(signals)} types")
        
        print(f"\n{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Prompt Analysis Pipeline")
    parser.add_argument("--simulation-dir", type=str, required=True,
                        help="Path to simulation directory to analyze")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output file for analysis results")
    parser.add_argument("--print-summary", action="store_true",
                        help="Print human-readable summary")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ComprehensivePromptAnalyzer(args.simulation_dir)
    
    # Run analysis
    print("🚀 Starting comprehensive prompt analysis...")
    results = analyzer.analyze_simulation()
    
    # Save results
    output_file = analyzer.save_analysis(args.output_file)
    
    # Print summary if requested
    if args.print_summary:
        analyzer.print_summary()
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
