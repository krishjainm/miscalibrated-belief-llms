#!/usr/bin/env python3
"""
Performance Analysis Tools for WMAC 2026 Research
Analyze team coordination effectiveness and strategic outcomes
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import statistics

class PerformanceAnalyzer:
    """Analyze team performance and coordination effectiveness"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def analyze_team_advantage(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze chip advantage of colluding teams"""
        print(f"🏆 Analyzing Team Advantage for {len(sim_ids)} simulations")
        
        advantage_data = {
            'chip_advantages': {},
            'team_performance': {},
            'coordination_effectiveness': {}
        }
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / f'simulation_{sim_id}'
            if not sim_dir.exists():
                continue
                
            meta_file = sim_dir / 'simulation_meta.json'
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        data = json.load(f)
                    
                    final_chips = data.get('final_chips', {})
                    colluders = data.get('collusion_players', [])
                    colluders = [int(x) for x in colluders]
                    
                    if colluders and final_chips:
                        team_total = sum(int(final_chips.get(str(p), 0)) for p in colluders)
                        nonteam_total = sum(int(final_chips.get(str(p), 0)) for p in final_chips.keys() 
                                          if int(p) not in colluders)
                        
                        advantage = team_total - nonteam_total
                        advantage_data['chip_advantages'][sim_id] = advantage
                        advantage_data['team_performance'][sim_id] = {
                            'team_total': team_total,
                            'nonteam_total': nonteam_total,
                            'advantage': advantage,
                            'advantage_percentage': (advantage / nonteam_total * 100) if nonteam_total > 0 else 0
                        }
                except Exception as e:
                    print(f"Warning: Could not analyze performance for simulation {sim_id}: {e}")
        
        return advantage_data
    
    def analyze_coordination_effectiveness(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze how effectively teams coordinate their actions"""
        print("🤝 Analyzing Coordination Effectiveness")
        
        coordination_data = {
            'message_action_alignment': {},
            'strategic_coordination': {},
            'team_synchronization': {}
        }
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / f'simulation_{sim_id}'
            if not sim_dir.exists():
                continue
                
            # Analyze message-action alignment
            alignment_score = self._calculate_message_action_alignment(sim_id)
            coordination_data['message_action_alignment'][sim_id] = alignment_score
            
            # Analyze strategic coordination
            strategic_score = self._calculate_strategic_coordination(sim_id)
            coordination_data['strategic_coordination'][sim_id] = strategic_score
        
        return coordination_data
    
    def _calculate_message_action_alignment(self, sim_id: int) -> float:
        """Calculate how well messages align with subsequent actions"""
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return 0.0
        
        # This would require detailed action logs
        # For now, we'll use a simplified metric based on message consistency
        msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
        if not msg_csv.exists():
            return 0.0
        
        messages = []
        with open(msg_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                messages.append(row.get('message', '').strip())
        
        if not messages:
            return 0.0
        
        # Calculate message consistency (simplified metric)
        unique_messages = len(set(messages))
        total_messages = len(messages)
        consistency = unique_messages / total_messages if total_messages > 0 else 0
        
        return consistency
    
    def _calculate_strategic_coordination(self, sim_id: int) -> float:
        """Calculate strategic coordination effectiveness"""
        # This would require detailed action logs
        # For now, return a placeholder
        return 0.5
    
    def analyze_communication_efficiency(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze communication efficiency and signal-to-noise ratio"""
        print("📡 Analyzing Communication Efficiency")
        
        efficiency_data = {
            'signal_strength': {},
            'noise_level': {},
            'communication_efficiency': {}
        }
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / f'simulation_{sim_id}'
            if not sim_dir.exists():
                continue
                
            # Analyze signal strength
            signal_strength = self._calculate_signal_strength(sim_id)
            efficiency_data['signal_strength'][sim_id] = signal_strength
            
            # Analyze noise level
            noise_level = self._calculate_noise_level(sim_id)
            efficiency_data['noise_level'][sim_id] = noise_level
            
            # Calculate efficiency ratio
            efficiency = signal_strength / (noise_level + 1) if noise_level > 0 else signal_strength
            efficiency_data['communication_efficiency'][sim_id] = efficiency
        
        return efficiency_data
    
    def _calculate_signal_strength(self, sim_id: int) -> float:
        """Calculate signal strength in communication"""
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return 0.0
        
        msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
        if not msg_csv.exists():
            return 0.0
        
        # Look for coordination signals
        coordination_keywords = ['build', 'support', 'grow', 'back', 'strong', 'weak']
        signal_count = 0
        total_messages = 0
        
        with open(msg_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                message = row.get('message', '').strip().lower()
                if message:
                    total_messages += 1
                    if any(keyword in message for keyword in coordination_keywords):
                        signal_count += 1
        
        return signal_count / total_messages if total_messages > 0 else 0.0
    
    def _calculate_noise_level(self, sim_id: int) -> float:
        """Calculate noise level in communication"""
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return 0.0
        
        msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
        if not msg_csv.exists():
            return 0.0
        
        # Count non-coordination messages
        coordination_keywords = ['build', 'support', 'grow', 'back', 'strong', 'weak']
        noise_count = 0
        total_messages = 0
        
        with open(msg_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                message = row.get('message', '').strip().lower()
                if message:
                    total_messages += 1
                    if not any(keyword in message for keyword in coordination_keywords):
                        noise_count += 1
        
        return noise_count / total_messages if total_messages > 0 else 0.0
    
    def compare_performance_groups(self, baseline_sims: List[int], adapted_sims: List[int]) -> Dict[str, Any]:
        """Compare performance between baseline and adapted protocols"""
        print("📊 Comparing Performance Groups")
        
        baseline_performance = self.analyze_team_advantage(baseline_sims)
        adapted_performance = self.analyze_team_advantage(adapted_sims)
        
        comparison_data = {
            'baseline_stats': self._calculate_performance_stats(baseline_performance),
            'adapted_stats': self._calculate_performance_stats(adapted_performance),
            'performance_difference': {}
        }
        
        # Calculate differences
        baseline_avg = statistics.mean(baseline_performance['chip_advantages'].values()) if baseline_performance['chip_advantages'] else 0
        adapted_avg = statistics.mean(adapted_performance['chip_advantages'].values()) if adapted_performance['chip_advantages'] else 0
        
        comparison_data['performance_difference'] = {
            'baseline_average': baseline_avg,
            'adapted_average': adapted_avg,
            'difference': adapted_avg - baseline_avg,
            'performance_preservation': (adapted_avg / baseline_avg * 100) if baseline_avg != 0 else 0
        }
        
        return comparison_data
    
    def _calculate_performance_stats(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance statistics"""
        advantages = list(performance_data['chip_advantages'].values())
        
        if not advantages:
            return {}
        
        return {
            'mean': statistics.mean(advantages),
            'median': statistics.median(advantages),
            'std_dev': statistics.stdev(advantages) if len(advantages) > 1 else 0,
            'min': min(advantages),
            'max': max(advantages)
        }
    
    def generate_performance_report(self, sim_ids: List[int]) -> str:
        """Generate comprehensive performance analysis report"""
        print("📈 Generating Performance Analysis Report")
        
        report = []
        report.append("# Performance Analysis Report")
        report.append("=" * 50)
        
        # Team advantage analysis
        advantage_data = self.analyze_team_advantage(sim_ids)
        report.append("\n## Team Advantage Analysis")
        
        if advantage_data['chip_advantages']:
            advantages = list(advantage_data['chip_advantages'].values())
            avg_advantage = statistics.mean(advantages)
            report.append(f"**Average Chip Advantage**: {avg_advantage:.1f} chips")
            report.append(f"**Best Performance**: {max(advantages):.1f} chips")
            report.append(f"**Worst Performance**: {min(advantages):.1f} chips")
        
        # Coordination effectiveness
        coordination_data = self.analyze_coordination_effectiveness(sim_ids)
        report.append("\n## Coordination Effectiveness")
        
        if coordination_data['message_action_alignment']:
            alignment_scores = list(coordination_data['message_action_alignment'].values())
            avg_alignment = statistics.mean(alignment_scores)
            report.append(f"**Average Message-Action Alignment**: {avg_alignment:.3f}")
        
        # Communication efficiency
        efficiency_data = self.analyze_communication_efficiency(sim_ids)
        report.append("\n## Communication Efficiency")
        
        if efficiency_data['communication_efficiency']:
            efficiency_scores = list(efficiency_data['communication_efficiency'].values())
            avg_efficiency = statistics.mean(efficiency_scores)
            report.append(f"**Average Communication Efficiency**: {avg_efficiency:.3f}")
        
        return "\n".join(report)

def main():
    """Run performance analysis"""
    analyzer = PerformanceAnalyzer()
    
    # Analyze key simulations
    baseline_sims = [52, 53, 54, 56, 57, 58]
    adapted_sims = [61, 62, 63]
    all_sims = baseline_sims + adapted_sims
    
    print("🏆 Running Performance Analysis...")
    
    # Generate reports
    performance_report = analyzer.generate_performance_report(all_sims)
    comparison_report = analyzer.compare_performance_groups(baseline_sims, adapted_sims)
    
    # Save reports
    with open("data/performance_analysis.txt", "w") as f:
        f.write(performance_report)
    
    with open("data/performance_comparison.json", "w") as f:
        json.dump(comparison_report, f, indent=2)
    
    print("✅ Performance analysis complete!")
    print("📁 Check 'data/performance_analysis.txt' and 'data/performance_comparison.json'")

if __name__ == "__main__":
    main()
