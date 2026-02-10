#!/usr/bin/env python3
"""
Protocol Analysis Tools for WMAC 2026 Research
Specialized analysis of emergent communication protocols
"""

import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import re

class ProtocolAnalyzer:
    """Analyze emergent communication protocols in detail"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def analyze_message_action_coupling(self, sim_id: int) -> Dict[str, Any]:
        """Analyze how well messages predict subsequent actions"""
        print(f"🔗 Analyzing Message-Action Coupling for Simulation {sim_id}")
        
        sim_dir = self.data_dir / f'simulation_{sim_id}'
        if not sim_dir.exists():
            return {}
        
        # Read messages and actions
        messages = []
        actions = []
        
        msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
        if msg_csv.exists():
            with open(msg_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    messages.append({
                        'player': int(row.get('player_id', 0)),
                        'message': row.get('message', '').strip(),
                        'hand': int(row.get('hand_id', 0)),
                        'phase': row.get('phase', ''),
                        'timestamp': row.get('timestamp', '')
                    })
        
        # Analyze coupling patterns
        coupling_data = {
            'message_types': Counter(),
            'action_correlations': {},
            'coordination_signals': []
        }
        
        # Look for specific coordination patterns
        coordination_patterns = {
            'building': ['build', 'building', 'grow', 'growing'],
            'supporting': ['support', 'backing', 'back'],
            'weak_hand': ['weak', 'fold', 'preserve']
        }
        
        for msg in messages:
            message_lower = msg['message'].lower()
            
            # Categorize message types
            for pattern_type, keywords in coordination_patterns.items():
                if any(keyword in message_lower for keyword in keywords):
                    coupling_data['message_types'][pattern_type] += 1
                    coupling_data['coordination_signals'].append({
                        'player': msg['player'],
                        'message': msg['message'],
                        'type': pattern_type,
                        'hand': msg['hand'],
                        'phase': msg['phase']
                    })
        
        return coupling_data
    
    def analyze_protocol_evolution(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze how protocols evolve across simulations"""
        print(f"🔄 Analyzing Protocol Evolution across {len(sim_ids)} simulations")
        
        evolution_data = {
            'message_diversity': {},
            'semantic_consistency': {},
            'adaptation_patterns': {}
        }
        
        for sim_id in sim_ids:
            coupling_data = self.analyze_message_action_coupling(sim_id)
            
            if coupling_data:
                evolution_data['message_diversity'][sim_id] = len(coupling_data['message_types'])
                evolution_data['semantic_consistency'][sim_id] = len(coupling_data['coordination_signals'])
        
        return evolution_data
    
    def analyze_lexical_adaptation(self, baseline_sims: List[int], adapted_sims: List[int]) -> Dict[str, Any]:
        """Compare protocols before and after lexical constraints"""
        print("📝 Analyzing Lexical Adaptation")
        
        adaptation_data = {
            'baseline_patterns': self._extract_message_patterns(baseline_sims),
            'adapted_patterns': self._extract_message_patterns(adapted_sims),
            'adaptation_effectiveness': {}
        }
        
        # Compare patterns
        baseline_messages = set(adaptation_data['baseline_patterns'].keys())
        adapted_messages = set(adaptation_data['adapted_patterns'].keys())
        
        adaptation_data['adaptation_effectiveness'] = {
            'preserved_patterns': len(baseline_messages & adapted_messages),
            'new_patterns': len(adapted_messages - baseline_messages),
            'lost_patterns': len(baseline_messages - adapted_messages),
            'adaptation_rate': len(adapted_messages - baseline_messages) / len(baseline_messages) if baseline_messages else 0
        }
        
        return adaptation_data
    
    def _extract_message_patterns(self, sim_ids: List[int]) -> Dict[str, int]:
        """Extract message patterns from simulations"""
        patterns = Counter()
        
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
                            patterns[message] += 1
        
        return dict(patterns)
    
    def analyze_semantic_preservation(self, sim_ids: List[int]) -> Dict[str, Any]:
        """Analyze how well semantic meaning is preserved under constraints"""
        print("🧠 Analyzing Semantic Preservation")
        
        semantic_data = {
            'concept_mapping': {},
            'semantic_consistency': {},
            'meaning_preservation': {}
        }
        
        # Define semantic concepts
        concepts = {
            'aggression': ['build', 'grow', 'raise', 'strong', 'pot'],
            'support': ['support', 'back', 'back', 'call', 'follow'],
            'weakness': ['weak', 'fold', 'preserve', 'conservative']
        }
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / f'simulation_{sim_id}'
            if not sim_dir.exists():
                continue
                
            msg_csv = sim_dir / 'chat_dataset' / 'messages.csv'
            if msg_csv.exists():
                concept_usage = {concept: 0 for concept in concepts.keys()}
                
                with open(msg_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        message = row.get('message', '').strip().lower()
                        if message:
                            for concept, keywords in concepts.items():
                                if any(keyword in message for keyword in keywords):
                                    concept_usage[concept] += 1
                
                semantic_data['concept_mapping'][sim_id] = concept_usage
        
        return semantic_data
    
    def generate_protocol_report(self, sim_ids: List[int]) -> str:
        """Generate detailed protocol analysis report"""
        print("📊 Generating Protocol Analysis Report")
        
        report = []
        report.append("# Protocol Analysis Report")
        report.append("=" * 50)
        
        # Message-Action Coupling Analysis
        coupling_results = {}
        for sim_id in sim_ids:
            coupling_results[sim_id] = self.analyze_message_action_coupling(sim_id)
        
        report.append("\n## Message-Action Coupling Analysis")
        for sim_id, data in coupling_results.items():
            if data:
                report.append(f"\n### Simulation {sim_id}")
                report.append(f"**Message Types**: {dict(data['message_types'])}")
                report.append(f"**Coordination Signals**: {len(data['coordination_signals'])}")
        
        # Protocol Evolution
        evolution_data = self.analyze_protocol_evolution(sim_ids)
        report.append("\n## Protocol Evolution")
        report.append(f"**Message Diversity**: {evolution_data['message_diversity']}")
        report.append(f"**Semantic Consistency**: {evolution_data['semantic_consistency']}")
        
        return "\n".join(report)

def main():
    """Run protocol analysis"""
    analyzer = ProtocolAnalyzer()
    
    # Analyze key simulations
    baseline_sims = [52, 53, 54, 56, 57, 58]
    adapted_sims = [61, 62, 63]
    
    print("🔬 Running Protocol Analysis...")
    
    # Generate reports
    baseline_report = analyzer.generate_protocol_report(baseline_sims)
    adapted_report = analyzer.generate_protocol_report(adapted_sims)
    
    # Save reports
    with open("data/baseline_protocol_analysis.txt", "w") as f:
        f.write(baseline_report)
    
    with open("data/adapted_protocol_analysis.txt", "w") as f:
        f.write(adapted_report)
    
    print("✅ Protocol analysis complete!")
    print("📁 Check 'data/baseline_protocol_analysis.txt' and 'data/adapted_protocol_analysis.txt'")

if __name__ == "__main__":
    main()
