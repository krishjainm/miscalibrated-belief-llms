"""
LLM agent implementation for Texas Hold'em poker.
This module provides an agent that uses a Language Model to make decisions in a poker game.
"""
from prompt_specs.baseline_prompts_v1 import get_pre_game_prompt_baseline, get_decision_prompt_baseline
import os
import re
from typing import Tuple, Optional, Dict, Any
import openai
from dotenv import load_dotenv
from texasholdem.texasholdem.game.game import TexasHoldEm
from texasholdem.texasholdem.game.action_type import ActionType
from texasholdem.texasholdem.game.player_state import PlayerState
from texasholdem.texasholdem.card.card import Card
from texasholdem.texasholdem.game.hand_phase import HandPhase
import json
from transformers import AutoTokenizer, PreTrainedModel
from utils.safe_json_parse import safe_json_parse


# ----------------------------------------------------------------------------------
# LLMAgent
# ----------------------------------------------------------------------------------
# This agent can now work with either:
#   1. A string representing an OpenAI model name (the old behaviour)
#   2. A Hugging Face `PreTrainedModel` object (e.g. Llama-3) that exposes
#      `.generate(...)`. When given such an object we will create an associated
#      tokenizer on the fly and will use `.generate` instead of the OpenAI API.
# ----------------------------------------------------------------------------------


class LLMAgent:
    """
    An agent that uses a Language Model to make decisions in a poker game.
    """

    def __init__(self, model, tokenizer, api_key: Optional[str] = None):
        """
        Initialize the LLM agent.

        Args:
            model: The model name to use or a Hugging Face `PreTrainedModel` object
            api_key: The API key. If None, will try to get from .env file
        """
        load_dotenv()

        # Decide whether we are in Hugging Face or OpenAI mode
        self.is_hf = not isinstance(model, str)

        if self.is_hf:
            # Hugging Face model branch ------------------------------------
            if not isinstance(model, PreTrainedModel):
                raise TypeError(
                    "When passing a non-string model it must be a HuggingFace PreTrainedModel instance"
                )

            self.model = model.eval()  # put into eval mode just in case

            # Attempt to locate an appropriate tokenizer. We fall back to the
            # model's `config._name_or_path` which usually stores the repo ID or path.
            model_id_or_path = getattr(model.config, "_name_or_path", None)
            if model_id_or_path is None:
                raise ValueError(
                    "Unable to determine the pretrained model path/name for tokenizer loading"
                )

            self.tokenizer = tokenizer

        else:
            # OpenAI model branch -------------------------------------------
            self.model = model  # model name (e.g. "gpt-4")

            if api_key:
                openai.api_key = api_key
            elif "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]
            else:
                raise ValueError("API key not provided and not found in .env file")

            # Initialize the client
            self.client = openai.OpenAI()

    def _format_game_state(self, game: TexasHoldEm, player_id: int) -> str:
        """
        Format the current game state as a string for the LLM.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A string representation of the game state
        """
        # Get player's hand
        hand = game.get_hand(player_id)
        hand_str = ", ".join(
            [
                f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}"
                for card in hand
            ]
        )

        # Get community cards
        community_cards = game.board
        community_str = (
            ", ".join(
                [
                    f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}"
                    for card in community_cards
                ]
            )
            if community_cards
            else "None"
        )

        # Get pot information
        pot_amount = game._get_last_pot().get_total_amount()

        # Get current phase
        phase = game.hand_phase.name

        # Get betting information
        chips_to_call = game.chips_to_call(player_id)
        min_raise = game.min_raise()

        # Get player positions and chips
        positions_info = []
        num_players = len(game.players)

        # Define position names based on number of players
        position_names = {
            2: ["SB", "BB"],
            3: ["SB", "BB", "UTG"],
            4: ["SB", "BB", "UTG", "CO"],
            5: ["SB", "BB", "UTG", "MP", "CO"],
            6: ["SB", "BB", "UTG", "MP", "CO", "BTN"],
        }

        # Get position names for current number of players
        current_positions = position_names.get(
            num_players, [f"P{i}" for i in range(num_players)]
        )

        # Rotate positions based on button location
        btn_loc = game.btn_loc
        rotated_positions = current_positions[btn_loc:] + current_positions[:btn_loc]

        for pos in range(len(game.players)):
            player = game.players[pos]
            state = "Folded" if player.state == PlayerState.OUT else "Active"
            position_name = rotated_positions[pos]
            positions_info.append(
                f"Position {pos} ({position_name}): {player.chips} chips ({state})"
            )

        # Get complete betting history
        betting_history = []
        if game.hand_history:
            for hand_phase in [
                HandPhase.PREFLOP,
                HandPhase.FLOP,
                HandPhase.TURN,
                HandPhase.RIVER,
            ]:
                if hand_phase in game.hand_history and game.hand_history[hand_phase]:
                    betting_history.append(f"\n{hand_phase.name}:")
                    for action in game.hand_history[hand_phase].actions:  # type: ignore[attr-defined]
                        position_name = rotated_positions[action.player_id]
                        betting_history.append(
                            f"Position {action.player_id} ({position_name}): {action.action_type.name} {action.total if action.total else ''}"
                        )

        # Format the state
        state = f"""
Current game state:
- Your position: {player_id} ({rotated_positions[player_id]})
- Small blind position: {game.sb_loc} ({rotated_positions[game.sb_loc]})
- Big blind position: {game.bb_loc} ({rotated_positions[game.bb_loc]})
- Your hand: {hand_str}
- Community cards: {community_str}
- Current phase: {phase}
- Pot amount: {pot_amount}
- Your chips: {game.players[player_id].chips}
- Chips to call: {chips_to_call}
- Minimum raise: {min_raise}

Player positions and chips:
{chr(10).join(positions_info)}

Betting history:
{chr(10).join(betting_history) if betting_history else "No betting history yet"}
"""
        return state

    def _get_available_actions(
        self, game: TexasHoldEm, player_id: int
    ) -> Dict[ActionType, str]:
        """
        Get the available actions for the player.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A dictionary mapping action types to descriptions
        """
        moves = game.get_available_moves()
        actions = {}

        # Calculate pot-based betting suggestions
        pot_amount = game._get_last_pot().get_total_amount()
        min_raise = game.min_raise()
        player_chips = game.players[player_id].chips
        previous_bet = game.player_bet_amount(player_id)

        # Add all available actions from the MoveIterator
        for action_type in moves.action_types:
            if action_type == ActionType.CHECK:
                actions[ActionType.CHECK] = "Check (pass the action without betting)"
            elif action_type == ActionType.CALL:
                actions[ActionType.CALL] = (
                    f"Call (match the current bet of {game.chips_to_call(player_id)} chips)"
                )
            elif action_type == ActionType.RAISE:
                # Calculate bet sizes based on pot and previous bet
                bet_sizes = []

                # Pot-based bet sizes
                pot_percentages = [0.33, 0.5, 0.66, 1.25]
                for percentage in pot_percentages:
                    suggested_amount = int(pot_amount * percentage)
                    if min_raise <= suggested_amount <= player_chips:
                        bet_sizes.append(
                            f"{int(percentage * 100)}% of pot ({suggested_amount} chips)"
                        )

                # Previous bet multiplier
                if previous_bet > 0:
                    suggested_amount = int(previous_bet * 2.5)
                    if min_raise <= suggested_amount <= player_chips:
                        bet_sizes.append(
                            f"2.5x previous bet ({suggested_amount} chips)"
                        )

                # Add all-in if it would be less than 20% of the pot
                if player_chips >= min_raise:
                    remaining_chips = player_chips - min_raise
                    if remaining_chips < pot_amount * 0.2:
                        bet_sizes.append(f"All-in ({player_chips} chips)")
                    elif len(bet_sizes) == 0:  # If no other valid bets, add all-in
                        bet_sizes.append(f"All-in ({player_chips} chips)")

                actions[ActionType.RAISE] = (
                    f"Raise (increase the bet, minimum raise is {min_raise} chips, maximum is {player_chips} chips)\n"
                    f"Bet choices:\n" + "\n".join(f"- {size}" for size in bet_sizes)
                )
            elif action_type == ActionType.FOLD:
                actions[ActionType.FOLD] = "Fold (give up the hand and exit the pot)"
            elif action_type == ActionType.ALL_IN:
                actions[ActionType.ALL_IN] = f"All-in (bet all {player_chips} chips)"

        return actions

    def get_action(
        self, game: TexasHoldEm, player_id: int
    ) -> Tuple[ActionType, Optional[int], Optional[str]]:
        print(f"[DEBUG] Calling get_action for player {player_id}")

        """
        Get the next action from the LLM.

        Args:
            game: The Texas Hold'em game
            player_id: The ID of the player making the decision

        Returns:
            A tuple of (action_type, total, reason) where:
                - action_type is the type of action to take
                - total is the amount to raise to (if applicable)
                - reason is the explanation for the action (None for LLM agents)
        """
        # Format the game state for the LLM
        state_str = self._format_game_state(game, player_id)

        # Get available actions
        available_actions = self._get_available_actions(game, player_id)

        # Create the prompt for the LLM
       # Create clean baseline prompts

        pre_game_prompt = get_pre_game_prompt_baseline()
        decision_prompt = get_decision_prompt_baseline(state_str)

        try:
            if self.is_hf:
                # -------------------------------------------------------
                # Hugging Face generation
                # -------------------------------------------------------
                import torch  # type: ignore

                # Tokenise the prompt
                inputs = self.tokenizer(decision_prompt, return_tensors="pt").to(
                    self.model.device
                )

                # Generate
                print("[DEBUG] About to generate using Hugging Face model...")
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                print("[DEBUG] Finished generating text.")
                # Decode – we only want the newly generated tokens
                generated_text = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

                content = generated_text.strip()
                print(f"[LLM RAW OUTPUT] {content}")

            else:
                # -------------------------------------------------------
                # OpenAI chat completion (legacy)
                # -------------------------------------------------------
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                    {"role": "system", "content": pre_game_prompt},
                    {"role": "user", "content": decision_prompt},
                ],
                    temperature=0.7,
                    max_tokens=50,
                )

                # Get the response content and clean it
                content = response.choices[0].message.content.strip()  # type: ignore[union-attr]

            # Try to find JSON in the response if there's additional text
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]

            # Parse the JSON response
            action_json = safe_json_parse(content)

            if "action" not in action_json:
                print("Error: Missing 'action' field in LLM response")
                return ActionType.FOLD, None, None

        except Exception as e:
            print(f"Error getting action from LLM: {str(e)}")
            return ActionType.FOLD, None, None
