from typing import Dict, Any, Optional
from texasholdem.game.history import SettleHistory
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.card.card import Card

def card_to_dict(card: Card) -> Dict[str, Any]:
    # Map suit integers to string names
    suit_map = {
        1: "spades",
        2: "hearts",
        4: "diamonds",
        8: "clubs"
    }
    
    return {
        "rank": card.rank,
        "suit": suit_map[card.suit]
    }

def action_to_dict(player_id: int, action_type: ActionType, total: Optional[int] = None, value: Optional[int] = None) -> Dict[str, Any]:
    return {
        "player_id": player_id,
        "action_type": action_type.name,
        "total": total,
        "value": value
    }

def betting_round_to_dict(new_cards: list, actions: list) -> Dict[str, Any]:
    return {
        "new_cards": [card_to_dict(card) for card in new_cards],
        "actions": [action_to_dict(action.player_id, action.action_type, action.total, action.value) for action in actions]
    }

def prehand_to_dict(btn_loc: int, big_blind: int, small_blind: int, player_chips: dict, player_cards: dict) -> Dict[str, Any]:
    return {
        "btn_loc": btn_loc,
        "big_blind": big_blind,
        "small_blind": small_blind,
        "player_chips": player_chips,
        "player_cards": {
            str(player_id): [card_to_dict(card) for card in cards]
            for player_id, cards in player_cards.items()
        }
    }

def settle_to_dict(new_cards: list, pot_winners: dict) -> Dict[str, Any]:
    """
    Convert a SettleHistory object to a dictionary.
    
    Returns:
        Dict[str, Any]: A dictionary representation of the settle history
    """
    game = TexasHoldEm.get_current_game()
    final_chips = {str(i): game.players[i].chips for i in range(len(game.players))}
    
    return {
        "new_cards": [card_to_dict(card) for card in new_cards],
        "pot_winners": {
            str(pot_id): {
                "amount": amount,
                "best_rank": best_rank,
                "winners": winners
            }
            for pot_id, (amount, best_rank, winners) in pot_winners.items()
        },
        "final_chips": final_chips
    } 