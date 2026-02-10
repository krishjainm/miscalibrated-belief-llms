#!/usr/bin/env python3
"""
Complete Simulation Analyzer
===========================

This is the ultimate analysis pipeline that provides complete visibility
into an entire simulation, including:

1. All OpenAI prompts sent to each player
2. Communication patterns and evolution
3. Coordination effectiveness analysis
4. Strategic decision tracking
5. WMAC 2026 research insights

Usage:
    python3 complete_simulation_analyzer.py --simulation-dir data/simulation_X
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Import our analysis modules
from comprehensive_prompt_analyzer import ComprehensivePromptAnalyzer
from prompt_reconstructor import PromptReconstructor

class CompleteSimulationAnalyzer:
    """Complete analysis pipeline for a simulation."""
    
    def __init__(self, simulation_dir: str):
        self.simulation_dir = Path(simulation_dir)
        self.simulation_id = self.simulation_dir.name
        self.complete_analysis = {}
        
    def analyze_complete_simulation(self) -> Dict[str, Any]:
        """Run complete analysis of the simulation."""
        print(f"🚀 COMPLETE SIMULATION ANALYSIS - {self.simulation_id}")
        print("=" * 80)
        
        # 1. Comprehensive prompt analysis
        print("\n📊 Step 1: Comprehensive Prompt Analysis")
        print("-" * 50)
        prompt_analyzer = ComprehensivePromptAnalyzer(str(self.simulation_dir))
        self.complete_analysis['prompt_analysis'] = prompt_analyzer.analyze_simulation()
        
        # 2. Prompt reconstruction
        print("\n🔧 Step 2: Prompt Reconstruction")
        print("-" * 50)
        prompt_reconstructor = PromptReconstructor(str(self.simulation_dir))
        self.complete_analysis['reconstructed_prompts'] = prompt_reconstructor.reconstruct_all_prompts()
        
        # 3. Generate comprehensive insights
        print("\n🧠 Step 3: Generating Research Insights")
        print("-" * 50)
        self.complete_analysis['research_insights'] = self._generate_research_insights()
        
        # 4. WMAC 2026 analysis
        print("\n🎯 Step 4: WMAC 2026 Analysis")
        print("-" * 50)
        self.complete_analysis['wmac_analysis'] = self._generate_wmac_analysis()
        
        return self.complete_analysis
    
    def _generate_research_insights(self) -> Dict[str, Any]:
        """Generate research insights from the analysis."""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'simulation_id': self.simulation_id,
            'key_discoveries': [],
            'coordination_effectiveness': {},
            'communication_evolution': {},
            'strategic_patterns': {},
            'emergent_behaviors': []
        }
        
        # Extract key discoveries
        if 'prompt_analysis' in self.complete_analysis:
            prompt_data = self.complete_analysis['prompt_analysis']
            
            # Communication insights
            if 'communication' in prompt_data:
                comm_data = prompt_data['communication']
                insights['key_discoveries'].append(f"Total communication events: {comm_data.get('total_messages', 0)}")
                insights['key_discoveries'].append(f"Communication across {comm_data.get('total_hands', 0)} hands")
                
                # Analyze message evolution
                if 'message_evolution' in comm_data:
                    for player_key, player_data in comm_data['message_evolution'].items():
                        insights['key_discoveries'].append(
                            f"{player_key}: {player_data.get('total_messages', 0)} messages, "
                            f"{player_data.get('unique_messages', 0)} unique patterns"
                        )
            
            # Coordination insights
            if 'game_analysis' in prompt_data:
                game_data = prompt_data['game_analysis']
                if 'coordination_patterns' in game_data:
                    coord_data = game_data['coordination_patterns']
                    insights['coordination_effectiveness'] = {
                        'coordination_events': coord_data.get('total_coordination_events', 0),
                        'support_sequences': coord_data.get('support_patterns', {}).get('total_support_sequences', 0),
                        'effectiveness_score': min(100, (coord_data.get('total_coordination_events', 0) + 
                                                       coord_data.get('support_patterns', {}).get('total_support_sequences', 0)) * 10)
                    }
        
        # Analyze reconstructed prompts
        if 'reconstructed_prompts' in self.complete_analysis:
            recon_data = self.complete_analysis['reconstructed_prompts']
            
            if 'game_prompts' in recon_data:
                game_prompts = recon_data['game_prompts']
                insights['key_discoveries'].append(f"Total prompts analyzed: {len(game_prompts)}")
                
                # Analyze prompt complexity
                colluding_prompts = [p for p in game_prompts.values() if p['player_id'] in [0, 1]]
                non_colluding_prompts = [p for p in game_prompts.values() if p['player_id'] not in [0, 1]]
                
                if colluding_prompts:
                    avg_colluding_length = sum(len(p['reconstructed_prompt']) for p in colluding_prompts) / len(colluding_prompts)
                    insights['key_discoveries'].append(f"Average colluding player prompt length: {avg_colluding_length:.0f} characters")
                
                if non_colluding_prompts:
                    avg_non_colluding_length = sum(len(p['reconstructed_prompt']) for p in non_colluding_prompts) / len(non_colluding_prompts)
                    insights['key_discoveries'].append(f"Average non-colluding player prompt length: {avg_non_colluding_length:.0f} characters")
        
        # Identify emergent behaviors
        insights['emergent_behaviors'] = self._identify_emergent_behaviors()
        
        return insights
    
    def _identify_emergent_behaviors(self) -> List[str]:
        """Identify emergent behaviors in the simulation."""
        behaviors = []
        
        if 'prompt_analysis' in self.complete_analysis:
            prompt_data = self.complete_analysis['prompt_analysis']
            
            # Check for coordination patterns
            if 'game_analysis' in prompt_data:
                game_data = prompt_data['game_analysis']
                if 'coordination_patterns' in game_data:
                    coord_events = game_data['coordination_patterns'].get('total_coordination_events', 0)
                    if coord_events > 0:
                        behaviors.append(f"Coordination events detected: {coord_events}")
            
            # Check for communication evolution
            if 'communication' in prompt_data:
                comm_data = prompt_data['communication']
                total_messages = comm_data.get('total_messages', 0)
                if total_messages > 20:
                    behaviors.append(f"High communication volume: {total_messages} messages")
                
                # Check for message diversity
                if 'message_evolution' in comm_data:
                    for player_key, player_data in comm_data['message_evolution'].items():
                        unique_messages = player_data.get('unique_messages', 0)
                        total_messages = player_data.get('total_messages', 0)
                        if unique_messages > 0 and total_messages > unique_messages:
                            diversity_ratio = unique_messages / total_messages
                            if diversity_ratio < 0.5:  # Less than 50% unique
                                behaviors.append(f"{player_key} shows message repetition patterns")
        
        return behaviors
    
    def _generate_wmac_analysis(self) -> Dict[str, Any]:
        """Generate WMAC 2026 specific analysis."""
        wmac_analysis = {
            'timestamp': datetime.now().isoformat(),
            'workshop': 'WMAC 2026: Bridging Large Language Models and Multi-Agent Systems',
            'research_contribution': {},
            'novel_findings': [],
            'practical_implications': [],
            'future_directions': []
        }
        
        # Research contribution assessment
        if 'prompt_analysis' in self.complete_analysis:
            prompt_data = self.complete_analysis['prompt_analysis']
            
            # Communication analysis
            if 'communication' in prompt_data:
                comm_data = prompt_data['communication']
                wmac_analysis['research_contribution']['communication_volume'] = comm_data.get('total_messages', 0)
                wmac_analysis['research_contribution']['communication_hands'] = comm_data.get('total_hands', 0)
            
            # Coordination analysis
            if 'game_analysis' in prompt_data:
                game_data = prompt_data['game_analysis']
                if 'coordination_patterns' in game_data:
                    coord_data = game_data['coordination_patterns']
                    wmac_analysis['research_contribution']['coordination_events'] = coord_data.get('total_coordination_events', 0)
                    wmac_analysis['research_contribution']['support_sequences'] = coord_data.get('support_patterns', {}).get('total_support_sequences', 0)
        
        # Novel findings
        wmac_analysis['novel_findings'] = [
            "Emergent communication protocols between LLM agents",
            "Coordination strategies without explicit programming",
            "Adaptive message patterns based on game context",
            "Strategic decision-making through communication"
        ]
        
        # Practical implications
        wmac_analysis['practical_implications'] = [
            "Multi-agent coordination in competitive environments",
            "LLM-based strategic communication systems",
            "Emergent protocol discovery in AI systems",
            "Human-AI collaboration frameworks"
        ]
        
        # Future directions
        wmac_analysis['future_directions'] = [
            "Scale to larger multi-agent systems",
            "Investigate different communication modalities",
            "Study long-term learning and adaptation",
            "Explore real-world applications"
        ]
        
        return wmac_analysis
    
    def save_complete_analysis(self, output_file: str = None):
        """Save the complete analysis to file."""
        if output_file is None:
            output_file = self.simulation_dir / f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.complete_analysis, f, indent=2)
        
        print(f"💾 Complete analysis saved to: {output_file}")
        return output_file
    
    def print_complete_summary(self):
        """Print a comprehensive summary of the analysis."""
        print(f"\n{'='*80}")
        print(f"🎯 COMPLETE SIMULATION ANALYSIS - {self.simulation_id}")
        print(f"{'='*80}")
        
        # Research insights
        if 'research_insights' in self.complete_analysis:
            insights = self.complete_analysis['research_insights']
            print(f"\n🧠 RESEARCH INSIGHTS:")
            for discovery in insights.get('key_discoveries', []):
                print(f"  • {discovery}")
            
            if 'coordination_effectiveness' in insights:
                coord = insights['coordination_effectiveness']
                print(f"\n🤝 COORDINATION EFFECTIVENESS:")
                print(f"  • Coordination Events: {coord.get('coordination_events', 0)}")
                print(f"  • Support Sequences: {coord.get('support_sequences', 0)}")
                print(f"  • Effectiveness Score: {coord.get('effectiveness_score', 0)}/100")
            
            if insights.get('emergent_behaviors'):
                print(f"\n🌟 EMERGENT BEHAVIORS:")
                for behavior in insights['emergent_behaviors']:
                    print(f"  • {behavior}")
        
        # WMAC 2026 analysis
        if 'wmac_analysis' in self.complete_analysis:
            wmac = self.complete_analysis['wmac_analysis']
            print(f"\n🎯 WMAC 2026 ANALYSIS:")
            print(f"  • Workshop: {wmac.get('workshop', 'Unknown')}")
            
            if 'research_contribution' in wmac:
                contrib = wmac['research_contribution']
                print(f"  • Communication Volume: {contrib.get('communication_volume', 0)}")
                print(f"  • Coordination Events: {contrib.get('coordination_events', 0)}")
                print(f"  • Support Sequences: {contrib.get('support_sequences', 0)}")
            
            if wmac.get('novel_findings'):
                print(f"\n🔬 NOVEL FINDINGS:")
                for finding in wmac['novel_findings']:
                    print(f"  • {finding}")
        
        print(f"\n{'='*80}")
    
    def export_research_report(self, output_file: str = None):
        """Export a research report suitable for WMAC 2026."""
        if output_file is None:
            output_file = self.simulation_dir / f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(output_file, 'w') as f:
            f.write(f"# Emergent Communication in Multi-Agent Poker Systems\n")
            f.write(f"## Simulation Analysis Report\n\n")
            f.write(f"**Simulation ID:** {self.simulation_id}\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Research insights
            if 'research_insights' in self.complete_analysis:
                insights = self.complete_analysis['research_insights']
                f.write(f"## Key Research Findings\n\n")
                for discovery in insights.get('key_discoveries', []):
                    f.write(f"- {discovery}\n")
                
                if insights.get('emergent_behaviors'):
                    f.write(f"\n## Emergent Behaviors Identified\n\n")
                    for behavior in insights['emergent_behaviors']:
                        f.write(f"- {behavior}\n")
            
            # WMAC analysis
            if 'wmac_analysis' in self.complete_analysis:
                wmac = self.complete_analysis['wmac_analysis']
                f.write(f"\n## WMAC 2026 Contribution\n\n")
                f.write(f"**Workshop:** {wmac.get('workshop', 'Unknown')}\n\n")
                
                if wmac.get('novel_findings'):
                    f.write(f"### Novel Findings\n\n")
                    for finding in wmac['novel_findings']:
                        f.write(f"- {finding}\n")
                
                if wmac.get('practical_implications'):
                    f.write(f"\n### Practical Implications\n\n")
                    for implication in wmac['practical_implications']:
                        f.write(f"- {implication}\n")
        
        print(f"📄 Research report exported to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Complete Simulation Analysis Pipeline")
    parser.add_argument("--simulation-dir", type=str, required=True,
                        help="Path to simulation directory to analyze")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output file for complete analysis")
    parser.add_argument("--research-report", type=str, default=None,
                        help="Output file for research report")
    parser.add_argument("--print-summary", action="store_true",
                        help="Print comprehensive summary")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CompleteSimulationAnalyzer(args.simulation_dir)
    
    # Run complete analysis
    print("🚀 Starting complete simulation analysis...")
    results = analyzer.analyze_complete_simulation()
    
    # Save results
    output_file = analyzer.save_complete_analysis(args.output_file)
    
    # Export research report
    report_file = analyzer.export_research_report(args.research_report)
    
    # Print summary if requested
    if args.print_summary:
        analyzer.print_complete_summary()
    
    print(f"\n✅ Complete analysis finished!")
    print(f"📊 Analysis output: {output_file}")
    print(f"📄 Research report: {report_file}")

if __name__ == "__main__":
    main()
