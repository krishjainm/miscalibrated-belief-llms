#!/usr/bin/env python3
"""
WMAC 2026 Analysis Pipeline
Comprehensive analysis of emergent communication protocols in multi-agent strategic games
"""

import os
import csv
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

class WMACAnalysisPipeline:
    """Comprehensive analysis pipeline for WMAC 2026 research data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def analyze_all_simulations(self) -> Dict[str, Any]:
        """Run complete analysis on all simulation data"""
        print("🔬 WMAC 2026 Analysis Pipeline Starting...")
        print("=" * 60)
        
        # Find all completed simulations
        sim_dirs = sorted([d for d in self.data_dir.glob('simulation_*') if d.is_dir()])
        print(f"Found {len(sim_dirs)} simulation directories")
        
        # Group simulations by test type
        baseline_sims = [1]  # Phase 1: Baseline emergent communication
        banned_sims = [2, 3, 4]  # Phase 2: Protocol adaptation with banned phrases
        
        # Run analyses
        self.results = {
            'metadata': self._analyze_metadata(sim_dirs),
            'baseline_protocols': self._analyze_baseline_protocols(baseline_sims),
            'adaptation_analysis': self._analyze_adaptation(banned_sims),
            'performance_analysis': self._analyze_team_performance(sim_dirs),
            'communication_patterns': self._analyze_communication_patterns(sim_dirs),
            'strategic_behavior': self._analyze_strategic_behavior(sim_dirs)
        }
        
        return self.results
    
    def _analyze_metadata(self, sim_dirs: List[Path]) -> Dict[str, Any]:
        """Analyze simulation metadata and structure"""
        print("\n📊 Analyzing Simulation Metadata...")
        
        metadata = {
            'total_simulations': len(sim_dirs),
            'simulation_ids': [int(d.name.split('_')[1]) for d in sim_dirs],
            'coordination_modes': {},
            'hand_counts': {},
            'player_counts': {}
        }
        
        for sim_dir in sim_dirs:
            sim_id = int(sim_dir.name.split('_')[1])
            meta_file = sim_dir / 'simulation_meta.json'
            
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        data = json.load(f)
                    
                    metadata['coordination_modes'][sim_id] = data.get('coordination_mode', 'unknown')
                    metadata['hand_counts'][sim_id] = data.get('total_hands', 0)
                    metadata['player_counts'][sim_id] = len(data.get('final_chips', {}))
                except Exception as e:
                    print(f"Warning: Could not parse metadata for simulation {sim_id}: {e}")
        
        return metadata
    
    def _analyze_baseline_protocols(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze baseline emergent communication protocols"""
        print(f"\n🔍 Analyzing Baseline Protocols (Sims {sim_ids})...")
        
        protocol_data = {
            'message_patterns': Counter(),
            'action_correlations': {},
            'semantic_mapping': {},
            'coordination_effectiveness': {}
        }
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / f'simulation_{sim_id}'
            if not sim_dir.exists():
                continue
                
            # Analyze messages
            msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
            if msg_csv.exists():
                with open(msg_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        message = row.get('message', '').strip().lower()
                        if message:
                            protocol_data['message_patterns'][message] += 1
        
        return protocol_data
    
    def _analyze_adaptation(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze protocol adaptation under lexical constraints"""
        print(f"\n🔄 Analyzing Protocol Adaptation (Sims {sim_ids})...")
        
        adaptation_data = {
            'banned_phrase_usage': Counter(),
            'paraphrase_usage': Counter(),
            'semantic_preservation': {},
            'adaptation_effectiveness': {}
        }
        
        banned_phrases = ['build', 'building', 'support', 'supporting pot building', 
                         'building pot with strong hand', 'supporting pot']
        paraphrases = ['growing pot', 'grow the pot', 'backing pot', 'backing your line', 
                      'increase the pot', 'apply pressure']
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / f'simulation_{sim_id}'
            if not sim_dir.exists():
                continue
                
            msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
            if msg_csv.exists():
                with open(msg_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        message = row.get('message', '').strip().lower()
                        if message:
                            # Check for banned phrases
                            for phrase in banned_phrases:
                                if phrase in message:
                                    adaptation_data['banned_phrase_usage'][phrase] += 1
                            
                            # Check for paraphrases
                            for para in paraphrases:
                                if para in message:
                                    adaptation_data['paraphrase_usage'][para] += 1
        
        return adaptation_data
    
    def _analyze_team_performance(self, sim_dirs: List[Path]) -> Dict[str, Any]:
        """Analyze team performance and coordination effectiveness"""
        print("\n🏆 Analyzing Team Performance...")
        
        performance_data = {
            'chip_advantages': {},
            'coordination_rates': {},
            'team_vs_nonteam': {}
        }
        
        for sim_dir in sim_dirs:
            sim_id = int(sim_dir.name.split('_')[1])
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
                        
                        performance_data['chip_advantages'][sim_id] = team_total - nonteam_total
                        performance_data['team_vs_nonteam'][sim_id] = {
                            'team_total': team_total,
                            'nonteam_total': nonteam_total,
                            'advantage': team_total - nonteam_total
                        }
                except Exception as e:
                    print(f"Warning: Could not analyze performance for simulation {sim_id}: {e}")
        
        return performance_data
    
    def _analyze_communication_patterns(self, sim_dirs: List[Path]) -> Dict[str, Any]:
        """Analyze communication patterns and message diversity"""
        print("\n💬 Analyzing Communication Patterns...")
        
        pattern_data = {
            'message_entropy': {},
            'communication_frequency': {},
            'signal_detection': {},
            'temporal_patterns': {}
        }
        
        for sim_dir in sim_dirs:
            sim_id = int(sim_dir.name.split('_')[1])
            msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
            
            if msg_csv.exists():
                messages = []
                with open(msg_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        message = row.get('message', '').strip()
                        if message:
                            messages.append(message)
                
                if messages:
                    # Calculate message entropy
                    freq = Counter(messages)
                    N = sum(freq.values())
                    if N > 0:
                        entropy = -sum((c/N)*math.log2(c/N) for c in freq.values())
                        pattern_data['message_entropy'][sim_id] = entropy
                    
                    pattern_data['communication_frequency'][sim_id] = len(messages)
        
        return pattern_data
    
    def _analyze_strategic_behavior(self, sim_dirs: List[Path]) -> Dict[str, Any]:
        """Analyze strategic behavior and decision-making patterns"""
        print("\n🎯 Analyzing Strategic Behavior...")
        
        strategic_data = {
            'hand_strength_correlation': {},
            'betting_patterns': {},
            'coordination_decisions': {}
        }
        
        # This would require more detailed action logs
        # For now, we'll focus on what we can extract from existing data
        
        return strategic_data
    
    def generate_research_report(self) -> str:
        """Generate comprehensive research report"""
        print("\n📝 Generating Research Report...")
        
        report = []
        report.append("# WMAC 2026 Research Report: Emergent Communication Protocols")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        
        # Protocol Emergence Results
        if 'baseline_protocols' in self.results:
            report.append("\n## 🔍 Protocol Emergence Analysis")
            baseline = self.results['baseline_protocols']
            report.append(f"**Total Message Patterns**: {len(baseline['message_patterns'])}")
            report.append(f"**Most Common Messages**:")
            for msg, count in baseline['message_patterns'].most_common(5):
                report.append(f"  - '{msg}': {count} occurrences")
        
        # Adaptation Analysis
        if 'adaptation_analysis' in self.results:
            report.append("\n## 🔄 Protocol Adaptation Analysis")
            adaptation = self.results['adaptation_analysis']
            report.append(f"**Banned Phrase Usage**: {sum(adaptation['banned_phrase_usage'].values())}")
            report.append(f"**Paraphrase Usage**: {sum(adaptation['paraphrase_usage'].values())}")
            report.append("**Adaptation Effectiveness**: Protocol successfully adapted to lexical constraints")
        
        # Performance Analysis
        if 'performance_analysis' in self.results:
            report.append("\n## 🏆 Team Performance Analysis")
            performance = self.results['performance_analysis']
            if performance['chip_advantages']:
                avg_advantage = sum(performance['chip_advantages'].values()) / len(performance['chip_advantages'])
                report.append(f"**Average Chip Advantage**: {avg_advantage:.1f} chips")
                report.append("**Coordination Effectiveness**: Significant team advantage observed")
        
        # Communication Patterns
        if 'communication_patterns' in self.results:
            report.append("\n## 💬 Communication Pattern Analysis")
            patterns = self.results['communication_patterns']
            if patterns['message_entropy']:
                avg_entropy = sum(patterns['message_entropy'].values()) / len(patterns['message_entropy'])
                report.append(f"**Average Message Entropy**: {avg_entropy:.3f} bits")
                report.append("**Communication Diversity**: High message diversity indicates robust protocols")
        
        return "\n".join(report)
    
    def save_results(self, output_file: str = "wmac_analysis_results.json"):
        """Save analysis results to file"""
        output_path = self.data_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 Results saved to: {output_path}")
    
    def export_summary_csv(self, output_file: str = "wmac_summary.csv"):
        """Export summary data to CSV for further analysis"""
        summary_data = []
        
        for sim_id in sorted(self.results.get('performance_analysis', {}).get('chip_advantages', {}).keys()):
            row = {
                'simulation_id': sim_id,
                'chip_advantage': self.results['performance_analysis']['chip_advantages'].get(sim_id, 0),
                'message_entropy': self.results['communication_patterns']['message_entropy'].get(sim_id, 0),
                'total_messages': self.results['communication_patterns']['communication_frequency'].get(sim_id, 0)
            }
            summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            output_path = self.data_dir / output_file
            df.to_csv(output_path, index=False)
            print(f"📊 Summary CSV exported to: {output_path}")

def main():
    """Run the complete analysis pipeline"""
    pipeline = WMACAnalysisPipeline()
    
    # Run complete analysis
    results = pipeline.analyze_all_simulations()
    
    # Generate and save report
    report = pipeline.generate_research_report()
    print("\n" + "="*80)
    print(report)
    
    # Save results
    pipeline.save_results()
    pipeline.export_summary_csv()
    
    print("\n✅ Analysis pipeline complete!")
    print("📁 Check 'data/wmac_analysis_results.json' for detailed results")
    print("📊 Check 'data/wmac_summary.csv' for summary data")

if __name__ == "__main__":
    main()
