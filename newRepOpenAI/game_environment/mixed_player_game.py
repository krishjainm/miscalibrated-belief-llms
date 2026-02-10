print("✅ USING official-llm-poker-collusion-main version")

"""
Mixed player game implementation for Texas Hold'em poker.
This module provides a game where some players are controlled by LLMs and others are human-controlled.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print("[DEBUG] Forced path:", Path(__file__).resolve().parent.parent)
import os
import time
from typing import List, Dict, Optional, Tuple, Set, Union
from utils.simulation_logger import SimulationLogger
from utils.game_state_extractor import extract_complete_game_state
from dotenv import load_dotenv
from texasholdem.texasholdem.game.game import TexasHoldEm
#from texasholdem.texasholdem.gui.text_gui import TextGUI
from texasholdem.texasholdem.game.action_type import ActionType
from game_environment.llm_agent import LLMAgent
from game_environment.collusion_llm_agent import CollusionLLMAgent
from game_environment.preflop_strategy import load_preflop_chart, lookup_action
# Removed transformers imports - API only now
import traceback
import json
from datetime import datetime

class MixedPlayerGame:
    """
    A Texas Hold'em game where some players are controlled by LLMs and others are human-controlled.
    """

    def __init__(
    self,
    buyin: int = 500,
    big_blind: int = 5,
    small_blind: int = 2,
    max_players: int = 6,
    llm_player_ids: Optional[List[int]] = None,
    collusion_llm_player_ids: Optional[List[int]] = None,
    openai_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    num_hands: int = 10,
    logger: Optional[SimulationLogger] = None  # ✅ Updated to SimulationLogger
):

        """
        Initialize the mixed player game.

        Args:
            buyin: The amount of chips each player starts with
            big_blind: The big blind amount
            small_blind: The small blind amount
            max_players: The maximum number of players
            llm_player_ids: The IDs of players controlled by regular LLM. If None, players 0 and 1 will be LLM-controlled.
            collusion_llm_player_ids: The IDs of players controlled by collusion LLM. If None, no players will be collusion LLM-controlled.
            openai_model: The model name to use. If None, will try to get from .env file
            openai_api_key: The API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file
        load_dotenv()

        # Store OpenAI configuration
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
        
        print(f"[DEBUG] Using OpenAI model: {self.openai_model}")

        self.game = TexasHoldEm(
            buyin=buyin,
            big_blind=big_blind,
            small_blind=small_blind,
            max_players=max_players,
        )
        self.gui = None

        # Set up AI players
        if llm_player_ids is None:
            llm_player_ids = [0, 1, 2, 3, 4, 5]  # Make all players LLM-controlled

        self.llm_player_ids = set(llm_player_ids)
        self.collusion_llm_player_ids = set(collusion_llm_player_ids)
        self.human_player_ids = (
            set(range(max_players))
            - self.llm_player_ids
            - self.collusion_llm_player_ids
        )

        self.num_hands = num_hands
        self.logger = logger or SimulationLogger()  # ✅ Use SimulationLogger by default
        
        # Initialize AI agents
        self.ai_agents = {}
        
        # Initialize collusion agents
        if collusion_llm_player_ids:
            for player_id in collusion_llm_player_ids:
                try:
                    agent = CollusionLLMAgent(
                        model=self.openai_model,
                        tokenizer=None,  # Not needed for OpenAI
                        api_key=self.openai_api_key,
                        teammate_id=next((pid for pid in collusion_llm_player_ids if pid != player_id), None)
                    )
                    self.ai_agents[player_id] = agent
                    print(f"✅ Created collusion LLM agent for player {player_id}")
                except Exception as e:
                    print(f"❌ Failed to create collusion LLM agent for player {player_id}: {e}")
        
        # Initialize regular LLM agents
        for player_id in llm_player_ids:
            if player_id not in self.ai_agents:  # Don't overwrite collusion agents
                try:
                    agent = LLMAgent(
                        model=self.openai_model,
                        tokenizer=None,  # Not needed for OpenAI
                        api_key=self.openai_api_key
                    )
                    self.ai_agents[player_id] = agent
                    print(f"✅ Created regular LLM agent for player {player_id}")
                except Exception as e:
                    print(f"❌ Failed to create regular LLM agent for player {player_id}: {e}")

    def _is_ai_player(self, player_id: int) -> bool:
        """Check if a player is controlled by an AI agent."""
        return player_id in self.ai_agents

    def _get_ai_action(self, player_id: int) -> Tuple[ActionType, Optional[int], str]:
        """Get action from AI agent."""
        if player_id not in self.ai_agents:
            print(f"[ERROR] No AI agent found for player {player_id}")
            return ActionType.FOLD, None, "No AI agent"
        
        try:
            agent = self.ai_agents[player_id]
            action_type, total, reason = agent.get_action(self.game, player_id)
            return action_type, total, reason or "AI decision"
        except Exception as e:
            print(f"[ERROR] AI agent error for player {player_id}: {e}")
            return ActionType.FOLD, None, f"AI error: {str(e)}"

    def _get_human_action(self) -> Tuple[ActionType, Optional[int]]:
        """Get action from human player (placeholder)."""
        # For now, just fold
        return ActionType.FOLD, None

    def run_game(self):
        """
        Run the game until it's over.
        """
        error_message = None
        try:
            # ✅ Start simulation logging
            simulation_id = self.logger.start_simulation()
            print(f"🎮 Starting simulation {simulation_id}")
            
            num_hands_played = 0
            while self.game.is_game_running() and num_hands_played < self.num_hands:
                player_chips_before = {p.player_id: p.chips for p in self.game.players}

                print(f"[DEBUG] Starting hand {num_hands_played + 1}...")
                self.game.start_hand()
                hand_id = self.game.get_hand_id()

                print(f"[DEBUG] Hand running? {self.game.is_hand_running()}")

                while self.game.is_hand_running():
                    current_player = self.game.current_player
                    print(f"[DEBUG] Current player: {current_player}")

                    # ✅ Extract complete game state before action
                    game_state = extract_complete_game_state(self.game, current_player)

                    if self._is_ai_player(current_player):
                        # Get action from AI
                        result = self._get_ai_action(current_player)
                        
                        if result is None:
                            print(f"[ERROR] Agent returned None. Forcing fold.")
                            action_type, total, reason = ActionType.FOLD, None, None
                        else:
                            action_type, total, reason = result
                            print(f"[DEBUG] ActionType: {action_type}, Total: {total}, Reason: {reason}")

                        # ✅ Log the action with complete game state
                        self.logger.log_action(
                            hand_id=hand_id,
                            phase=game_state["phase"],
                            player_id=current_player,
                            action_type=action_type.name,
                            amount=total,
                            reason=reason,
                            game_state=game_state
                        )

                        # Take the action
                        try:
                            if action_type == ActionType.RAISE and total is not None:
                                print(f"[DEBUG] Taking RAISE action with total={total}")
                                self.game.take_action(action_type, total=total)
                            else:
                                print(f"[DEBUG] Taking action: {action_type}")
                                self.game.take_action(action_type)

                        except Exception as e:
                            print(f"[ERROR] Action failed: {e}. Forcing fold.")
                            self.game.take_action(ActionType.FOLD)

                    else:
                        # Get action from human
                        self._get_human_action()
                
                # ✅ Force settlement if needed
                if self.game.hand_phase != HandPhase.SETTLE:
                    print("[WARNING] Forcing hand settlement.")
                    self.game.settle_hand()

                # ✅ Extract winner and pot after settlement
                try:
                    winning_player = self.game.get_winner()
                    pot_size = self.game._get_last_pot().get_total_amount()
                    
                    # Calculate chip differences
                    player_chips_after = {p.player_id: p.chips for p in self.game.players}
                    chip_diff = {
                        pid: player_chips_after[pid] - player_chips_before.get(pid, 0)
                        for pid in player_chips_after
                    }
                    
                    # Log hand summary
                    hand_summary = {
                        "winner": winning_player,
                        "pot": pot_size,
                        "chip_diff": chip_diff,
                        "final_chips": player_chips_after
                    }
                    
                    self.logger.log_hand_summary(hand_id, hand_summary)
                    print(f"✅ Hand {hand_id} complete. Winner: {winning_player}, Pot: {pot_size}")
                    
                except Exception as e:
                    print(f"[WARNING] Could not determine winner or pot: {e}")

                num_hands_played += 1
                time.sleep(1)

            # ✅ End simulation with final stats
            final_stats = {
                "total_hands": num_hands_played,
                "final_chips": {p.player_id: p.chips for p in self.game.players},
                "collusion_players": list(self.collusion_llm_player_ids),
                "llm_players": list(self.llm_player_ids),
                "human_players": list(self.human_player_ids),
                "coordination_mode": getattr(self, 'coordination_mode', None)
            }
            
            self.logger.end_simulation(final_stats)
            print(f"✅ Simulation {simulation_id} completed!")

        except Exception as e:
            # Save the error message and include full traceback
            error_message = f"\nError occurred: {str(e)}\n{traceback.format_exc()}"
        else:
            # No error occurred
            error_message = None
        finally:
            # Always clean up the curses session
            #self.gui.hide()
            # Reset the terminal
            # os.system("reset")  # Commented out because 'reset' is for Unix, not Windows

            # Display the error message after cleanup if there was one
            if error_message:
                print(error_message)


if __name__ == "__main__":
    import argparse
    from utils.simulation_logger import SimulationLogger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-hands", type=int, default=10)
    args = parser.parse_args()

    logger = SimulationLogger()

    game = MixedPlayerGame(
        buyin=500,
        big_blind=1,
        small_blind=2,
        max_players=2,
        llm_player_ids=[],
        collusion_llm_player_ids=[0, 1],
        openai_model="gpt-4o",
        openai_api_key=None,
        num_hands=args.num_hands,
        logger=logger 
    )

    game.run_game()
