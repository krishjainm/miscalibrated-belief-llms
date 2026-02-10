#!/usr/bin/env python3
"""
Comprehensive Comparison Test: newRepOpenAI vs Original System
Runs 10 simulations each with 20 hands to compare performance
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

def run_newrepopenai_tests():
    """Run 10 simulations with newRepOpenAI system"""
    print("🧪 Running newRepOpenAI System Tests")
    print("=" * 50)
    
    results = []
    
    for i in range(1, 11):
        print(f"\n📊 Simulation {i}/10 - newRepOpenAI")
        print("-" * 30)
        
        start_time = time.time()
        
        # Run the simulation
        cmd = [
            "python3", "run_prompt_inspector.py",
            "--num-hands", "20",
            "--model", "gpt-3.5-turbo", 
            "--collusion-llm-players", "0,1",
            "--communication-style", "emergent",
            "--coordination-mode", "emergent_only"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse the results from the simulation metadata
                sim_dir = f"data/simulation_{i + 12}"  # Offset to avoid conflicts
                meta_file = Path(sim_dir) / "simulation_meta.json"
                
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                    
                    final_chips = meta['final_stats']['final_chips']
                    colluding_players = meta['final_stats']['collusion_players']
                    
                    # Calculate results
                    colluding_total = sum(final_chips[str(p)] for p in colluding_players)
                    non_colluding_total = sum(final_chips[str(p)] for p in [2, 3])
                    
                    result_data = {
                        'simulation_id': i,
                        'system': 'newRepOpenAI',
                        'final_chips': final_chips,
                        'colluding_total': colluding_total,
                        'non_colluding_total': non_colluding_total,
                        'colluding_advantage': colluding_total - non_colluding_total,
                        'duration_seconds': time.time() - start_time,
                        'success': True
                    }
                    
                    print(f"✅ Colluding: {colluding_total}, Non-colluding: {non_colluding_total}")
                    print(f"📈 Advantage: {result_data['colluding_advantage']:+d} chips")
                    
                else:
                    result_data = {
                        'simulation_id': i,
                        'system': 'newRepOpenAI', 
                        'success': False,
                        'error': 'No metadata file found'
                    }
                    print("❌ No metadata file found")
                    
            else:
                result_data = {
                    'simulation_id': i,
                    'system': 'newRepOpenAI',
                    'success': False,
                    'error': result.stderr
                }
                print(f"❌ Command failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            result_data = {
                'simulation_id': i,
                'system': 'newRepOpenAI',
                'success': False,
                'error': 'Timeout after 5 minutes'
            }
            print("⏰ Timeout after 5 minutes")
        except Exception as e:
            result_data = {
                'simulation_id': i,
                'system': 'newRepOpenAI',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Exception: {e}")
        
        results.append(result_data)
        time.sleep(2)  # Brief pause between simulations
    
    return results

def run_original_system_tests():
    """Run 10 simulations with original pokerWork-updates07 system"""
    print("\n🧪 Running Original System Tests")
    print("=" * 50)
    
    results = []
    
    for i in range(1, 11):
        print(f"\n📊 Simulation {i}/10 - Original System")
        print("-" * 30)
        
        start_time = time.time()
        
        # Run the original system (we'll need to implement this)
        # For now, let's simulate the results based on what we know
        result_data = {
            'simulation_id': i,
            'system': 'Original',
            'success': False,
            'error': 'Original system test not implemented yet'
        }
        print("⚠️ Original system test not implemented yet")
        
        results.append(result_data)
        time.sleep(2)
    
    return results

def analyze_results(newrepopenai_results, original_results):
    """Analyze and compare the results"""
    print("\n📊 COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Filter successful results
    newrepopenai_success = [r for r in newrepopenai_results if r['success']]
    original_success = [r for r in original_results if r['success']]
    
    print(f"\nnewRepOpenAI System:")
    print(f"  Successful simulations: {len(newrepopenai_success)}/10")
    if newrepopenai_success:
        avg_advantage = sum(r['colluding_advantage'] for r in newrepopenai_success) / len(newrepopenai_success)
        print(f"  Average colluding advantage: {avg_advantage:+.1f} chips")
        
        wins = len([r for r in newrepopenai_success if r['colluding_advantage'] > 0])
        print(f"  Colluding players won: {wins}/{len(newrepopenai_success)} simulations")
    
    print(f"\nOriginal System:")
    print(f"  Successful simulations: {len(original_success)}/10")
    if original_success:
        avg_advantage = sum(r['colluding_advantage'] for r in original_success) / len(original_success)
        print(f"  Average colluding advantage: {avg_advantage:+.1f} chips")
        
        wins = len([r for r in original_success if r['colluding_advantage'] > 0])
        print(f"  Colluding players won: {wins}/{len(original_success)} simulations")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comparison_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'newrepopenai_results': newrepopenai_results,
            'original_results': original_results,
            'analysis': {
                'newrepopenai_success_rate': len(newrepopenai_success) / 10,
                'original_success_rate': len(original_success) / 10,
                'newrepopenai_avg_advantage': sum(r['colluding_advantage'] for r in newrepopenai_success) / len(newrepopenai_success) if newrepopenai_success else 0,
                'original_avg_advantage': sum(r['colluding_advantage'] for r in original_success) / len(original_success) if original_success else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")

def main():
    print("🚀 COMPREHENSIVE SYSTEM COMPARISON TEST")
    print("=" * 60)
    print("Testing: newRepOpenAI vs Original pokerWork-updates07")
    print("Configuration: 10 simulations × 20 hands each")
    print("Colluding players: 0, 1 | Non-colluding: 2, 3")
    print("=" * 60)
    
    # Run newRepOpenAI tests
    newrepopenai_results = run_newrepopenai_tests()
    
    # Run original system tests (placeholder for now)
    original_results = run_original_system_tests()
    
    # Analyze results
    analyze_results(newrepopenai_results, original_results)
    
    print("\n✅ Comparison test completed!")

if __name__ == "__main__":
    main()
