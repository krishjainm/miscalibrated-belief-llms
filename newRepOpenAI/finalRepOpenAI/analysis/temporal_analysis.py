#!/usr/bin/env python3
"""
Temporal Analysis for WMAC 2026 Research
Analyze communication patterns over time and game phases
"""

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import statistics
from datetime import datetime

class TemporalAnalyzer:
    """Analyze temporal patterns in communication and coordination"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def analyze_communication_timing(self, sim_id: int) -> Dict[str, Any]:
        """Analyze when messages occur relative to game phases"""
        print(f"⏰ Analyzing Communication Timing for Simulation {sim_id}")
        
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return {}
        
        timing_data = {
            'phase_distribution': Counter(),
            'hand_communication_density': {},
            'message_intervals': [],
            'coordination_timing': {}
        }
        
        msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
        if not msg_csv.exists():
            return timing_data
        
        messages_by_hand = defaultdict(list)
        phase_messages = defaultdict(list)
        
        with open(msg_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hand_id = int(row.get('hand_id', 0))
                phase = row.get('phase', '')
                message = row.get('message', '').strip()
                timestamp = row.get('timestamp', '')
                
                if message:
                    messages_by_hand[hand_id].append({
                        'phase': phase,
                        'message': message,
                        'timestamp': timestamp,
                        'player': int(row.get('player_id', 0))
                    })
                    phase_messages[phase].append(message)
                    timing_data['phase_distribution'][phase] += 1
        
        # Analyze communication density per hand
        for hand_id, messages in messages_by_hand.items():
            timing_data['hand_communication_density'][hand_id] = len(messages)
        
        # Calculate message intervals (simplified)
        all_timestamps = [msg['timestamp'] for hand_msgs in messages_by_hand.values() 
                         for msg in hand_msgs if msg['timestamp']]
        if len(all_timestamps) > 1:
            # This would need proper timestamp parsing for real intervals
            timing_data['message_intervals'] = [1] * (len(all_timestamps) - 1)  # Placeholder
        
        return timing_data
    
    def analyze_coordination_sequences(self, sim_id: int) -> Dict[str, Any]:
        """Analyze sequences of coordination between teammates"""
        print(f"🔄 Analyzing Coordination Sequences for Simulation {sim_id}")
        
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return {}
        
        sequence_data = {
            'coordination_patterns': [],
            'response_times': [],
            'coordination_effectiveness': {}
        }
        
        msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
        if not msg_csv.exists():
            return sequence_data
        
        # Group messages by hand and analyze sequences
        hand_messages = defaultdict(list)
        
        with open(msg_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hand_id = int(row.get('hand_id', 0))
                player_id = int(row.get('player_id', 0))
                message = row.get('message', '').strip()
                phase = row.get('phase', '')
                
                if message:
                    hand_messages[hand_id].append({
                        'player': player_id,
                        'message': message,
                        'phase': phase
                    })
        
        # Analyze coordination patterns
        for hand_id, messages in hand_messages.items():
            if len(messages) >= 2:  # Need at least 2 messages for coordination
                # Look for coordination patterns
                coordination_signals = []
                for msg in messages:
                    if any(keyword in msg['message'].lower() for keyword in 
                          ['build', 'support', 'grow', 'back', 'strong', 'weak']):
                        coordination_signals.append(msg)
                
                if len(coordination_signals) >= 2:
                    sequence_data['coordination_patterns'].append({
                        'hand_id': hand_id,
                        'signals': len(coordination_signals),
                        'players': [msg['player'] for msg in coordination_signals]
                    })
        
        return sequence_data
    
    def analyze_phase_communication(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze communication patterns across different game phases"""
        print(f"🎯 Analyzing Phase Communication for {len(sim_ids)} simulations")
        
        phase_data = {
            'phase_frequency': Counter(),
            'phase_effectiveness': {},
            'communication_density': {}
        }
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / f'simulation_{sim_id}'
            if not sim_dir.exists():
                continue
            
            msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
            if not msg_csv.exists():
                continue
            
            phase_messages = defaultdict(list)
            
            with open(msg_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    phase = row.get('phase', '')
                    message = row.get('message', '').strip()
                    
                    if message:
                        phase_messages[phase].append(message)
                        phase_data['phase_frequency'][phase] += 1
            
            # Calculate communication density per phase
            for phase, messages in phase_messages.items():
                if phase not in phase_data['communication_density']:
                    phase_data['communication_density'][phase] = []
                phase_data['communication_density'][phase].append(len(messages))
        
        return phase_data
    
    def analyze_hand_by_hand_performance(self, sim_id: int) -> Dict[str, Any]:
        """Analyze performance and communication on a hand-by-hand basis"""
        print(f"📊 Analyzing Hand-by-Hand Performance for Simulation {sim_id}")
        
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return {}
        
        hand_data = {
            'hand_communication': {},
            'hand_outcomes': {},
            'coordination_success': {}
        }
        
        # Read hand summaries if available
        hand_summaries = list(sim_dir.glob('hand_*_summary.json'))
        for summary_file in hand_summaries:
            try:
                with open(summary_file) as f:
                    hand_info = json.load(f)
                hand_id = hand_info.get('hand_id', 0)
                hand_data['hand_outcomes'][hand_id] = hand_info
            except Exception:
                continue
        
        # Read messages and correlate with hands
        msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
        if msg_csv.exists():
            hand_messages = defaultdict(list)
            
            with open(msg_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hand_id = int(row.get('hand_id', 0))
                    message = row.get('message', '').strip()
                    
                    if message:
                        hand_messages[hand_id].append(message)
            
            for hand_id, messages in hand_messages.items():
                hand_data['hand_communication'][hand_id] = {
                    'message_count': len(messages),
                    'coordination_signals': sum(1 for msg in messages 
                                              if any(keyword in msg.lower() 
                                                    for keyword in ['build', 'support', 'grow', 'back']))
                }
        
        return hand_data
    
    def generate_temporal_report(self, sim_ids: List[int]) -> str:
        """Generate comprehensive temporal analysis report"""
        print("📈 Generating Temporal Analysis Report")
        
        report = []
        report.append("# Temporal Communication Analysis Report")
        report.append("=" * 60)
        
        # Phase communication analysis
        phase_data = self.analyze_phase_communication(sim_ids)
        report.append("\n## Phase Communication Analysis")
        report.append(f"**Phase Distribution**: {dict(phase_data['phase_frequency'])}")
        
        # Communication density by phase
        if phase_data['communication_density']:
            report.append("\n### Communication Density by Phase")
            for phase, densities in phase_data['communication_density'].items():
                avg_density = statistics.mean(densities) if densities else 0
                report.append(f"**{phase}**: {avg_density:.1f} messages per simulation")
        
        # Individual simulation analysis
        report.append("\n## Individual Simulation Analysis")
        for sim_id in sim_ids:
            timing_data = self.analyze_communication_timing(sim_id)
            sequence_data = self.analyze_coordination_sequences(sim_id)
            hand_data = self.analyze_hand_by_hand_performance(sim_id)
            
            report.append(f"\n### Simulation {sim_id}")
            if timing_data:
                report.append(f"**Phase Distribution**: {dict(timing_data['phase_distribution'])}")
                report.append(f"**Hand Communication Density**: {len(timing_data['hand_communication_density'])} hands with communication")
            
            if sequence_data:
                report.append(f"**Coordination Patterns**: {len(sequence_data['coordination_patterns'])} coordinated sequences")
            
            if hand_data:
                report.append(f"**Hand Communication**: {len(hand_data['hand_communication'])} hands analyzed")
        
        return "\n".join(report)

def main():
    """Run temporal analysis"""
    analyzer = TemporalAnalyzer()
    
    # Analyze key simulations
    baseline_sims = [52, 53, 54, 56, 57, 58]
    adapted_sims = [61, 62, 63]
    all_sims = baseline_sims + adapted_sims
    
    print("⏰ Running Temporal Analysis...")
    
    # Generate report
    temporal_report = analyzer.generate_temporal_report(all_sims)
    
    # Save report
    with open("data/temporal_analysis.txt", "w") as f:
        f.write(temporal_report)
    
    print("✅ Temporal analysis complete!")
    print("📁 Check 'data/temporal_analysis.txt' for results")

if __name__ == "__main__":
    main()
