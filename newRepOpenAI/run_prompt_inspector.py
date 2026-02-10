#!/usr/bin/env python3
"""
OpenAI Prompt Inspector - Run poker games with complete prompt visibility
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
from dotenv import load_dotenv
from prompt_logger import PromptLogger

# Import the modified game environment
from game_environment.mixed_player_communication_game import MixedPlayerCommunicationGame
from game_environment.advanced_collusion_agent import AdvancedCollusionAgent
from game_environment.communicating_llm_agent import CommunicatingLLMAgent

# Load environment variables
load_dotenv()

class PromptInspectorGame(MixedPlayerCommunicationGame):
    """Modified game class that logs all OpenAI prompts"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_logger = PromptLogger()
        
    def start_simulation(self, simulation_id: str):
        """Start a new simulation with prompt logging"""
        # Get attributes and convert sets to lists for JSON serialization
        llm_player_ids = getattr(self, 'llm_player_ids', [])
        collusion_llm_player_ids = getattr(self, 'collusion_llm_player_ids', [])
        communication_config = getattr(self, 'communication_config', {})
        
        # Convert sets to lists if needed
        if isinstance(llm_player_ids, set):
            llm_player_ids = list(llm_player_ids)
        if isinstance(collusion_llm_player_ids, set):
            collusion_llm_player_ids = list(collusion_llm_player_ids)
        
        config = {
            "num_hands": getattr(self, 'num_hands', 10),
            "buyin": getattr(self, 'buyin', 500),
            "big_blind": getattr(self, 'big_blind', 5),
            "small_blind": getattr(self, 'small_blind', 2),
            "max_players": getattr(self, 'max_players', 6),
            "llm_player_ids": llm_player_ids,
            "collusion_llm_player_ids": collusion_llm_player_ids,
            "communication_config": communication_config
        }
        
        self.prompt_logger.start_simulation(simulation_id, config)
        print(f"🔍 Prompt logging started for simulation {simulation_id}")
        
    def _log_llm_interaction(self, player_id: int, phase: str, prompt: str, 
                           response: str, response_time_ms: int = None):
        """Log LLM interaction with full context"""
        
        # Get current game state
        game_state = {
            "hand_id": getattr(self, 'current_hand', 0),
            "phase": phase,
            "player_id": player_id,
            "pot_size": getattr(self.game, 'pot', 0),
            "player_chips": getattr(self.game, 'players', [{}])[player_id].get('chips', 0) if hasattr(self.game, 'players') else 0,
            "board_cards": getattr(self.game, 'board', []) if hasattr(self.game, 'board') else [],
            "hole_cards": getattr(self.game, 'players', [{}])[player_id].get('hole_cards', []) if hasattr(self.game, 'players') else []
        }
        
        # Get chat history if available
        chat_history = getattr(self, 'chat_history', [])
        
        # Log the prompt
        self.prompt_logger.log_prompt(
            player_id=player_id,
            phase=phase,
            prompt=prompt,
            model=getattr(self, 'openai_model', 'gpt-3.5-turbo'),
            temperature=0.7,  # Default temperature
            max_tokens=150,   # Default max tokens
            game_state=game_state,
            chat_history=chat_history,
            response=response,
            response_time_ms=response_time_ms
        )
        
        print(f"📝 Logged prompt for Player {player_id} in {phase}")
    
    def _create_llm_agent_with_logging(self, player_id: int, is_colluding: bool = False):
        """Create LLM agent with prompt logging"""
        
        if is_colluding:
            agent = AdvancedCollusionAgent(
                player_id=player_id,
                model=getattr(self, 'openai_model', 'gpt-3.5-turbo'),
                api_key=getattr(self, 'openai_api_key', None),
                temperature=0.7,
                max_tokens=150
            )
            
            # NEW: Set up teammate coordination
            collusion_players = getattr(self, 'collusion_llm_player_ids', [])
            if isinstance(collusion_players, set):
                collusion_players = list(collusion_players)
            
            # Find the other colluding player as teammate
            teammates = [p for p in collusion_players if p != player_id]
            print(f"[DEBUG] Player {player_id} collusion_players: {collusion_players}, teammates: {teammates}")
            if teammates:
                agent.set_teammate(teammates[0])  # Set first teammate
                print(f"[COORDINATION SETUP] Player {player_id} teammate set to {teammates[0]}")
            else:
                print(f"[COORDINATION WARNING] Player {player_id} has no teammates!")
        else:
            agent = CommunicatingLLMAgent(
                player_id=player_id,
                model=getattr(self, 'openai_model', 'gpt-3.5-turbo'),
                api_key=getattr(self, 'openai_api_key', None),
                temperature=0.7,
                max_tokens=150
            )
        
        # Wrap the agent's methods to log prompts
        original_get_action = agent.get_action
        
        def logged_get_action(game_state, available_actions, chat_history=None):
            # Get the prompt that would be sent
            prompt = agent._build_prompt(game_state, available_actions, chat_history)
            
            # Call original method
            import time
            start_time = time.time()
            response = original_get_action(game_state, available_actions, chat_history)
            response_time = int((time.time() - start_time) * 1000)
            
            # Log the interaction
            self._log_llm_interaction(
                player_id=player_id,
                phase=getattr(game_state, 'phase', 'UNKNOWN'),
                prompt=prompt,
                response=response,
                response_time_ms=response_time
            )
            
            return response
        
        agent.get_action = logged_get_action
        return agent

def main():
    parser = argparse.ArgumentParser(description="Run poker games with OpenAI prompt logging")
    parser.add_argument("--num-hands", type=int, default=3, help="Number of hands to play")
    parser.add_argument("--buyin", type=int, default=500, help="Starting chips per player")
    parser.add_argument("--big-blind", type=int, default=5, help="Big blind amount")
    parser.add_argument("--small-blind", type=int, default=2, help="Small blind amount")
    parser.add_argument("--max-players", type=int, default=4, help="Maximum number of players")
    parser.add_argument("--llm-players", type=str, default="0,1,2,3", help="Comma-separated LLM player IDs")
    parser.add_argument("--collusion-llm-players", type=str, default="0,1", help="Comma-separated colluding player IDs")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--communication-level", type=str, default="moderate", 
                       choices=["none", "limited", "moderate", "full"], 
                       help="Communication level")
    parser.add_argument("--communication-style", type=str, default="emergent",
                       choices=["cooperative", "emergent", "steganographic_self", "steganographic_guided", "subtle", "deceptive", "emergent_discovery", "emergent_adaptive", "emergent_exploratory"],
                       help="Communication style")
    parser.add_argument("--collusion-strategy", type=str, default="signal_and_squeeze",
                       choices=["signal_and_squeeze", "chip_dumping", "information_sharing", "whipsaw"],
                       help="Collusion strategy")
    parser.add_argument("--coordination-mode", type=str, default="explicit",
                       choices=["explicit", "advisory", "emergent_only"],
                       help="Coordination mode: explicit (overrides), advisory (no overrides), emergent_only (disabled)")
    parser.add_argument("--output-dir", type=str, default="prompt_inspector_output", help="Output directory")
    parser.add_argument("--log-prompts", action="store_true", help="Enable prompt logging")
    parser.add_argument("--view-prompts", action="store_true", help="View prompts after game")
    
    args = parser.parse_args()
    
    # Parse player lists
    llm_players = [int(x.strip()) for x in args.llm_players.split(",")]
    collusion_llm_players = [int(x.strip()) for x in args.collusion_llm_players.split(",")]
    
    print("🔍 OpenAI Prompt Inspector")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Hands: {args.num_hands}")
    print(f"Players: {args.max_players}")
    print(f"LLM Players: {llm_players}")
    print(f"Colluding Players: {collusion_llm_players}")
    print(f"Communication Level: {args.communication_level}")
    print(f"Communication Style: {args.communication_style}")
    print(f"Prompt Logging: {'Enabled' if args.log_prompts else 'Disabled'}")
    print(f"Coordination Mode: {args.coordination_mode}")
    print("=" * 50)
    
    # Create game instance
    game = PromptInspectorGame(
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
            "strategy": args.collusion_strategy,
            "coordination_mode": args.coordination_mode
        }
    )
    
    # Start simulation with prompt logging
    if args.log_prompts:
        game.start_simulation("prompt_inspector_1")
    
    # Run the game
    print("\n🎮 Starting game with prompt logging...")
    game.run_game()
    
    # Save session data
    if args.log_prompts:
        game.prompt_logger.save_session()
        print(f"\n📊 Prompt logging complete!")
        print(f"📁 Logs saved to: {game.prompt_logger.output_dir}")
        print(f"📈 Total prompts logged: {len(game.prompt_logger.prompts)}")
        
        # Export to CSV
        csv_file = game.prompt_logger.export_to_csv()
        print(f"📄 CSV export: {csv_file}")
    
    # View prompts if requested
    if args.view_prompts and args.log_prompts:
        print("\n🔍 Recent prompts:")
        for prompt in game.prompt_logger.get_recent_prompts(5):
            print(f"\n--- Player {prompt['player_id']} ({prompt['phase']}) ---")
            print(f"Prompt: {prompt['prompt'][:200]}...")
            if prompt['response']:
                print(f"Response: {prompt['response'][:100]}...")

if __name__ == "__main__":
    main()
