#!/usr/bin/env python3
"""
Run poker game with enhanced prompt logging to see exact OpenAI inputs
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_prompt_logger import EnhancedPromptLogger
from run_prompt_inspector import PromptInspectorGame

class EnhancedLoggingGame(PromptInspectorGame):
    """Game with enhanced prompt logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enhanced_logger = EnhancedPromptLogger()
    
    def start_simulation(self, simulation_id: str):
        """Start simulation with enhanced logging"""
        super().start_simulation(simulation_id)
        
        # Get simulation config
        config = {
            "num_hands": getattr(self, 'num_hands', 10),
            "buyin": getattr(self, 'buyin', 500),
            "big_blind": getattr(self, 'big_blind', 5),
            "small_blind": getattr(self, 'small_blind', 2),
            "max_players": getattr(self, 'max_players', 6),
            "llm_player_ids": getattr(self, 'llm_player_ids', []),
            "collusion_llm_player_ids": getattr(self, 'collusion_llm_player_ids', []),
            "communication_config": getattr(self, 'communication_config', {}),
            "coordination_mode": getattr(self, 'coordination_mode', 'emergent_only')
        }
        
        self.enhanced_logger.start_simulation(simulation_id, config)
        print(f"🔍 Enhanced prompt logging started for simulation {simulation_id}")
    
    def _log_llm_interaction(self, player_id: int, phase: str, prompt: str, 
                           response: str, response_time_ms: int, game_state: dict, 
                           chat_history: list = None):
        """Log complete LLM interaction with enhanced logging"""
        
        # Get model details
        model = getattr(self, 'openai_model', 'gpt-3.5-turbo')
        temperature = 0.7  # Default
        max_tokens = 150   # Default
        
        # Parse response to get action details
        action_taken = None
        action_amount = None
        
        try:
            import json
            if response.strip().startswith('{'):
                parsed = json.loads(response)
                action_taken = parsed.get('action')
                action_amount = parsed.get('amount', 0)
        except:
            pass
        
        # Log with enhanced logger
        self.enhanced_logger.log_complete_interaction(
            player_id=player_id,
            phase=phase,
            full_prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            game_state=game_state,
            chat_history=chat_history,
            response=response,
            response_time_ms=response_time_ms,
            action_taken=action_taken,
            action_amount=action_amount
        )
        
        # Also log with original logger for compatibility
        super()._log_llm_interaction(player_id, phase, prompt, response, response_time_ms, game_state, chat_history)
    
    def end_simulation(self):
        """End simulation and save enhanced logs"""
        self.enhanced_logger.save_session()
        super().end_simulation()

def main():
    parser = argparse.ArgumentParser(description="Run poker game with enhanced prompt logging")
    parser.add_argument("--num-hands", type=int, default=10, help="Number of hands to play")
    parser.add_argument("--buyin", type=int, default=500, help="Starting chips")
    parser.add_argument("--big-blind", type=int, default=5, help="Big blind amount")
    parser.add_argument("--small-blind", type=int, default=2, help="Small blind amount")
    parser.add_argument("--max-players", type=int, default=6, help="Maximum players")
    parser.add_argument("--llm-players", type=str, default="0,1,2,3", help="LLM player IDs (comma-separated)")
    parser.add_argument("--collusion-llm-players", type=str, default="0,1", help="Colluding LLM player IDs (comma-separated)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--communication-level", type=str, default="moderate", 
                        choices=["none", "limited", "moderate", "full"],
                        help="Communication level")
    parser.add_argument("--communication-style", type=str, default="emergent",
                        choices=["cooperative", "emergent", "steganographic_self", "steganographic_guided", "subtle", "deceptive"],
                        help="Communication style")
    parser.add_argument("--collusion-strategy", type=str, default="signal_and_squeeze",
                        choices=["signal_and_squeeze", "coordination", "information_sharing"],
                        help="Collusion strategy")
    parser.add_argument("--coordination-mode", type=str, default="emergent_only",
                        choices=["explicit", "advisory", "emergent_only"],
                        help="Coordination mode")
    
    args = parser.parse_args()
    
    # Parse player IDs
    llm_players = [int(x.strip()) for x in args.llm_players.split(",")]
    collusion_llm_players = [int(x.strip()) for x in args.collusion_llm_players.split(",")]
    
    # Create game with enhanced logging
    game = EnhancedLoggingGame(
        num_hands=args.num_hands,
        buyin=args.buyin,
        big_blind=args.big_blind,
        small_blind=args.small_blind,
        max_players=args.max_players,
        llm_player_ids=llm_players,
        collusion_llm_player_ids=collusion_llm_players,
        openai_model=args.model,
        openai_api_key=args.api_key,
        communication_config={
            "level": args.communication_level,
            "style": args.communication_style,
            "strategy": args.collusion_strategy
        },
        coordination_mode=args.coordination_mode
    )
    
    print("🚀 Starting poker game with enhanced prompt logging...")
    print(f"📊 Configuration:")
    print(f"   - Hands: {args.num_hands}")
    print(f"   - Model: {args.model}")
    print(f"   - Communication: {args.communication_style}")
    print(f"   - Collusion: {args.collusion_strategy}")
    print(f"   - Coordination: {args.coordination_mode}")
    print(f"   - LLM Players: {llm_players}")
    print(f"   - Colluding Players: {collusion_llm_players}")
    print()
    
    # Run the game
    game.run_game()
    
    print("\n✅ Game completed with enhanced prompt logging!")
    print(f"📁 Logs saved to: data/enhanced_prompt_logs/")

if __name__ == "__main__":
    main()
