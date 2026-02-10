#!/usr/bin/env python3
"""
Live Metrics Analysis for WMAC 2026 Research
Analyze real-time metrics that were captured during simulations
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import statistics

class LiveMetricsAnalyzer:
    """Analyze live metrics captured during simulations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def extract_live_metrics(self, sim_id: int) -> Dict[str, Any]:
        """Extract live metrics from simulation logs"""
        print(f"📊 Extracting Live Metrics for Simulation {sim_id}")
        
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return {}
        
        live_metrics = {
            'communication_stats': {},
            'coordination_debug': [],
            'action_validation': [],
            'hand_summaries': []
        }
        
        # Look for any log files that might contain live metrics
        log_files = list(sim_dir.glob('*.log')) + list(sim_dir.glob('*.txt'))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Extract communication statistics
                comm_stats_match = re.search(r'Total messages: (\d+)', content)
                if comm_stats_match:
                    live_metrics['communication_stats']['total_messages'] = int(comm_stats_match.group(1))
                
                unique_speakers_match = re.search(r'Unique speakers: (\d+)', content)
                if unique_speakers_match:
                    live_metrics['communication_stats']['unique_speakers'] = int(unique_speakers_match.group(1))
                
                avg_length_match = re.search(r'Average message length: ([\d.]+)', content)
                if avg_length_match:
                    live_metrics['communication_stats']['avg_message_length'] = float(avg_length_match.group(1))
                
                signals_match = re.search(r'Potential signals detected: (\d+)', content)
                if signals_match:
                    live_metrics['communication_stats']['signals_detected'] = int(signals_match.group(1))
                
                # Extract coordination debug info
                coord_debug_matches = re.findall(r'\[COORDINATION DEBUG\].*', content)
                live_metrics['coordination_debug'].extend(coord_debug_matches)
                
                # Extract action validation info
                validation_matches = re.findall(r'\[DEBUG VALIDATION\].*', content)
                live_metrics['action_validation'].extend(validation_matches)
                
                # Extract hand summaries
                hand_summary_matches = re.findall(r'📊 Logged hand summary: hand_\d+_summary\.json', content)
                live_metrics['hand_summaries'].extend(hand_summary_matches)
                
            except Exception as e:
                print(f"Warning: Could not read {log_file}: {e}")
        
        return live_metrics
    
    def analyze_coordination_effectiveness(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze coordination effectiveness from live metrics"""
        print(f"🤝 Analyzing Coordination Effectiveness for {len(sim_ids)} simulations")
        
        effectiveness_data = {
            'signal_rates': {},
            'coordination_success': {},
            'communication_efficiency': {}
        }
        
        for sim_id in sim_ids:
            live_metrics = self.extract_live_metrics(sim_id)
            
            if live_metrics and 'communication_stats' in live_metrics:
                stats = live_metrics['communication_stats']
                
                if 'total_messages' in stats and 'signals_detected' in stats:
                    signal_rate = stats['signals_detected'] / stats['total_messages'] if stats['total_messages'] > 0 else 0
                    effectiveness_data['signal_rates'][sim_id] = signal_rate
                
                if 'total_messages' in stats and 'unique_speakers' in stats:
                    efficiency = stats['total_messages'] / stats['unique_speakers'] if stats['unique_speakers'] > 0 else 0
                    effectiveness_data['communication_efficiency'][sim_id] = efficiency
        
        return effectiveness_data
    
    def analyze_message_patterns(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze message patterns from live metrics"""
        print(f"💬 Analyzing Message Patterns for {len(sim_ids)} simulations")
        
        pattern_data = {
            'message_lengths': {},
            'signal_density': {},
            'communication_consistency': {}
        }
        
        for sim_id in sim_ids:
            live_metrics = self.extract_live_metrics(sim_id)
            
            if live_metrics and 'communication_stats' in live_metrics:
                stats = live_metrics['communication_stats']
                
                if 'avg_message_length' in stats:
                    pattern_data['message_lengths'][sim_id] = stats['avg_message_length']
                
                if 'signals_detected' in stats and 'total_messages' in stats:
                    density = stats['signals_detected'] / stats['total_messages'] if stats['total_messages'] > 0 else 0
                    pattern_data['signal_density'][sim_id] = density
        
        return pattern_data
    
    def analyze_simulation_progression(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze how simulations progress over time"""
        print(f"📈 Analyzing Simulation Progression for {len(sim_ids)} simulations")
        
        progression_data = {
            'completion_rates': {},
            'communication_evolution': {},
            'performance_trends': {}
        }
        
        for sim_id in sim_ids:
            live_metrics = self.extract_live_metrics(sim_id)
            
            # Check if simulation completed
            if live_metrics and 'hand_summaries' in live_metrics:
                progression_data['completion_rates'][sim_id] = len(live_metrics['hand_summaries'])
            
            # Analyze communication evolution
            if live_metrics and 'coordination_debug' in live_metrics:
                coord_events = len(live_metrics['coordination_debug'])
                progression_data['communication_evolution'][sim_id] = coord_events
        
        return progression_data
    
    def generate_live_metrics_report(self, sim_ids: List[int]) -> str:
        """Generate comprehensive live metrics report"""
        print("📊 Generating Live Metrics Report")
        
        report = []
        report.append("# Live Metrics Analysis Report")
        report.append("=" * 60)
        
        # Coordination effectiveness
        effectiveness_data = self.analyze_coordination_effectiveness(sim_ids)
        report.append("\n## Coordination Effectiveness Analysis")
        
        if effectiveness_data['signal_rates']:
            avg_signal_rate = statistics.mean(effectiveness_data['signal_rates'].values())
            report.append(f"**Average Signal Rate**: {avg_signal_rate:.3f}")
        
        if effectiveness_data['communication_efficiency']:
            avg_efficiency = statistics.mean(effectiveness_data['communication_efficiency'].values())
            report.append(f"**Average Communication Efficiency**: {avg_efficiency:.1f} messages per speaker")
        
        # Message patterns
        pattern_data = self.analyze_message_patterns(sim_ids)
        report.append("\n## Message Pattern Analysis")
        
        if pattern_data['message_lengths']:
            avg_length = statistics.mean(pattern_data['message_lengths'].values())
            report.append(f"**Average Message Length**: {avg_length:.1f} characters")
        
        if pattern_data['signal_density']:
            avg_density = statistics.mean(pattern_data['signal_density'].values())
            report.append(f"**Average Signal Density**: {avg_density:.3f}")
        
        # Simulation progression
        progression_data = self.analyze_simulation_progression(sim_ids)
        report.append("\n## Simulation Progression Analysis")
        
        if progression_data['completion_rates']:
            avg_completion = statistics.mean(progression_data['completion_rates'].values())
            report.append(f"**Average Completion Rate**: {avg_completion:.1f} hands")
        
        if progression_data['communication_evolution']:
            avg_evolution = statistics.mean(progression_data['communication_evolution'].values())
            report.append(f"**Average Coordination Events**: {avg_evolution:.1f}")
        
        # Individual simulation details
        report.append("\n## Individual Simulation Details")
        for sim_id in sim_ids:
            live_metrics = self.extract_live_metrics(sim_id)
            report.append(f"\n### Simulation {sim_id}")
            
            if live_metrics and 'communication_stats' in live_metrics:
                stats = live_metrics['communication_stats']
                report.append(f"**Total Messages**: {stats.get('total_messages', 'N/A')}")
                report.append(f"**Unique Speakers**: {stats.get('unique_speakers', 'N/A')}")
                report.append(f"**Average Message Length**: {stats.get('avg_message_length', 'N/A')}")
                report.append(f"**Signals Detected**: {stats.get('signals_detected', 'N/A')}")
            
            if live_metrics and 'coordination_debug' in live_metrics:
                report.append(f"**Coordination Events**: {len(live_metrics['coordination_debug'])}")
        
        return "\n".join(report)

def main():
    """Run live metrics analysis"""
    analyzer = LiveMetricsAnalyzer()
    
    # Analyze key simulations
    baseline_sims = [52, 53, 54, 56, 57, 58]
    adapted_sims = [61, 62, 63]
    all_sims = baseline_sims + adapted_sims
    
    print("📊 Running Live Metrics Analysis...")
    
    # Generate report
    live_report = analyzer.generate_live_metrics_report(all_sims)
    
    # Save report
    with open("data/live_metrics_analysis.txt", "w") as f:
        f.write(live_report)
    
    print("✅ Live metrics analysis complete!")
    print("📁 Check 'data/live_metrics_analysis.txt' for results")

if __name__ == "__main__":
    main()
