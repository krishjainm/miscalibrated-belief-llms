from typing import Dict, Any, List
from texasholdem.texasholdem.game.game import TexasHoldEm
from texasholdem.texasholdem.card.card import Card

def extract_complete_game_state(game: TexasHoldEm, current_player: int) -> Dict[str, Any]:
    """
    Extract complete game state for logging.
    Includes all players' hands, chips, positions, and current game context.
    """
    
    # Get current phase
    phase = game.hand_phase.name.lower()
    
    # Get pot information
    pot_amount = game._get_last_pot().get_total_amount()
    
    # Get community cards
    community_cards = []
    if game.board:
        community_cards = [
            {
                "rank": card.rank,
                "suit": card.suit,
                "str_rank": Card.STR_RANKS[card.rank],
                "str_suit": Card.INT_SUIT_TO_CHAR_SUIT[card.suit],
                "display": f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}"
            }
            for card in game.board
        ]
    
    # Get all players' information
    players_info = {}
    for player in game.players:
        player_id = player.player_id
        
        # Get player's hand (if they have cards)
        hand_cards = []
        try:
            hand = game.get_hand(player_id)
            hand_cards = [
                {
                    "rank": card.rank,
                    "suit": card.suit,
                    "str_rank": Card.STR_RANKS[card.rank],
                    "str_suit": Card.INT_SUIT_TO_CHAR_SUIT[card.suit],
                    "display": f"{Card.STR_RANKS[card.rank]}{Card.INT_SUIT_TO_CHAR_SUIT[card.suit]}"
                }
                for card in hand
            ]
        except:
            # Player might not have cards (folded, etc.)
            pass
        
        # Get player's position
        position_name = get_position_name(player_id, game.btn_loc, len(game.players))
        
        players_info[player_id] = {
            "chips": player.chips,
            "state": player.state.name,
            "position": position_name,
            "hand_cards": hand_cards,
            "is_current_player": player_id == current_player,
            "is_button": player_id == game.btn_loc,
            "is_small_blind": player_id == game.sb_loc,
            "is_big_blind": player_id == game.bb_loc
        }
    
    # Get betting information
    chips_to_call = game.chips_to_call(current_player) if current_player is not None else 0
    min_raise = game.min_raise()
    
    # Get betting history for current hand
    betting_history = []
    if hasattr(game, 'hand_history') and game.hand_history:
        for phase_name in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']:
            phase_history = getattr(game.hand_history, phase_name.lower(), None)
            if phase_history and hasattr(phase_history, 'actions'):
                phase_actions = []
                for action in phase_history.actions:
                    phase_actions.append({
                        "player_id": action.player_id,
                        "action_type": action.action_type.name,
                        "amount": action.total,
                        "position": get_position_name(action.player_id, game.btn_loc, len(game.players))
                    })
                if phase_actions:
                    betting_history.append({
                        "phase": phase_name.lower(),
                        "actions": phase_actions
                    })
    
    # Compile complete game state
    game_state = {
        "hand_id": game.get_hand_id(),
        "current_player": current_player,
        "phase": phase,
        "pot_amount": pot_amount,
        "chips_to_call": chips_to_call,
        "min_raise": min_raise,
        "community_cards": community_cards,
        "players": players_info,
        "betting_history": betting_history,
        "button_position": game.btn_loc,
        "small_blind_position": game.sb_loc,
        "big_blind_position": game.bb_loc,
        "num_players": len(game.players),
        "num_active_players": len([p for p in game.players if p.state.name != "OUT"])
    }
    
    return game_state

def get_position_name(player_id: int, btn_loc: int, num_players: int) -> str:
    """Get position name (SB, BB, UTG, etc.) for a player."""
    
    position_names = {
        2: ["SB", "BB"],
        3: ["SB", "BB", "UTG"],
        4: ["SB", "BB", "UTG", "CO"],
        5: ["SB", "BB", "UTG", "MP", "CO"],
        6: ["SB", "BB", "UTG", "MP", "CO", "BTN"],
    }
    
    current_positions = position_names.get(num_players, [f"P{i}" for i in range(num_players)])
    
    # Rotate positions based on button location
    rotated_positions = current_positions[btn_loc:] + current_positions[:btn_loc]
    
    return rotated_positions[player_id] if player_id < len(rotated_positions) else f"P{player_id}" 