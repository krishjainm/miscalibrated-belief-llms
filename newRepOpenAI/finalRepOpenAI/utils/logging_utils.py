import os
import json
from datetime import datetime
from typing import List, Dict, Any, Union

class HandHistoryLogger:
    def __init__(self, log_dir: str = "data/hand_history"):
        print("[DEBUG] ✅ USING logging_utils.py from official-llm-poker-collusion-main")
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.history: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]):
        self.history.append(record)
        
    print(f"[DEBUG] ACTUALLY USING THIS FILE: {__file__}")
    
    def log_hand(self, hand_log: dict, hand_id: Union[int, str]):
        print(f"[DEBUG] Using HandHistoryLogger from: {__file__}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_{hand_id}_summary_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        print(f"[DEBUG] Writing log to: {filepath}")
        print(f"[DEBUG] About to dump JSON: {hand_log}")  # ✅ Debug

        with open(filepath, "w") as f:
            json.dump(hand_log, f, indent=2)

        print(f"[DEBUG] ✅ Successfully wrote: {filepath}")
    
    def log_action(self, action_log: dict, hand_id: Union[int, str], action_num: int):
        """
        Log individual actions during the game for real-time analysis.
        
        Args:
            action_log: Dictionary containing action details
            hand_id: Current hand ID
            action_num: Sequential action number within the hand
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_{hand_id}_action_{action_num:03d}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(action_log, f, indent=2)
        
        print(f"[DEBUG] ✅ Logged action {action_num} for hand {hand_id}")
    
    def log_game_state(self, game_state: dict, hand_id: Union[int, str], phase: str):
        """
        Log the current game state before each action.
        
        Args:
            game_state: Dictionary containing current game state
            hand_id: Current hand ID  
            phase: Current game phase (preflop, flop, turn, river)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_{hand_id}_state_{phase}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(game_state, f, indent=2)
        
        print(f"[DEBUG] ✅ Logged game state for hand {hand_id} at {phase}")