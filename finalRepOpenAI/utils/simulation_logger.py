import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class SimulationLogger:
    """
    Hierarchical logging system for poker simulations.
    Creates simulation_1, simulation_2, etc. folders with complete game state logging.
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.simulation_dir = None
        self.simulation_id = None
        self.game_logs_dir = None
        self.action_counter = 0
        
    def start_simulation(self) -> str:
        """
        Start a new simulation and create the folder structure.
        Returns the simulation ID.
        """
        # Find next available simulation number
        simulation_id = 1
        while (self.base_dir / f"simulation_{simulation_id}").exists():
            simulation_id += 1
            
        self.simulation_id = simulation_id
        self.simulation_dir = self.base_dir / f"simulation_{simulation_id}"
        self.game_logs_dir = self.simulation_dir / "game_logs"
        
        # Create directory structure
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        self.game_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simulation metadata
        simulation_meta = {
            "simulation_id": simulation_id,
            "start_time": datetime.now().isoformat(),
            "description": f"Poker simulation {simulation_id}",
            "status": "running"
        }
        
        with open(self.simulation_dir / "simulation_meta.json", "w") as f:
            json.dump(simulation_meta, f, indent=2)
            
        print(f"🎮 Started simulation {simulation_id} at {self.simulation_dir}")
        return str(simulation_id)
    
    def log_action(self, 
                   hand_id: int,
                   phase: str,
                   player_id: int,
                   action_type: str,
                   amount: Optional[int],
                   reason: Optional[str],
                   game_state: Dict[str, Any]) -> None:
        """
        Log a complete action with full game state.
        
        Args:
            hand_id: Current hand number
            phase: Game phase (preflop, flop, turn, river)
            player_id: Player making the action
            action_type: Type of action (CALL, FOLD, RAISE, etc.)
            amount: Bet amount (if any)
            reason: Reason for action (from LLM)
            game_state: Complete game state dictionary
        """
        if not self.simulation_dir:
            raise ValueError("Simulation not started. Call start_simulation() first.")
            
        # Create action log with complete context
        action_log = {
            "timestamp": datetime.now().isoformat(),
            "hand_id": hand_id,
            "phase": phase,
            "player_id": player_id,
            "action_type": action_type,
            "amount": amount,
            "reason": reason,
            "action_number": self.action_counter,
            "game_state": game_state
        }
        
        # Create filename: hand_1_preflop_player_0_call.json
        filename = f"hand_{hand_id}_{phase}_player_{player_id}_{action_type.lower()}.json"
        filepath = self.game_logs_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(action_log, f, indent=2)
            
        self.action_counter += 1
        print(f"📝 Logged action: {filename}")
    
    def log_hand_summary(self, hand_id: int, hand_summary: Dict[str, Any]) -> None:
        """
        Log a hand summary at the end of each hand.
        """
        if not self.simulation_dir:
            raise ValueError("Simulation not started. Call start_simulation() first.")
            
        filename = f"hand_{hand_id}_summary.json"
        filepath = self.game_logs_dir / filename
        
        hand_summary["timestamp"] = datetime.now().isoformat()
        hand_summary["hand_id"] = hand_id
        
        with open(filepath, "w") as f:
            json.dump(hand_summary, f, indent=2)
            
        print(f"📊 Logged hand summary: {filename}")
    
    def end_simulation(self, final_stats: Dict[str, Any]) -> None:
        """
        End the simulation and log final statistics.
        """
        if not self.simulation_dir:
            raise ValueError("Simulation not started. Call start_simulation() first.")
            
        # Update simulation metadata
        simulation_meta = {
            "simulation_id": self.simulation_id,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "description": f"Poker simulation {self.simulation_id}",
            "status": "completed",
            "total_actions": self.action_counter,
            "final_stats": final_stats
        }
        
        with open(self.simulation_dir / "simulation_meta.json", "w") as f:
            json.dump(simulation_meta, f, indent=2)
            
        print(f"✅ Completed simulation {self.simulation_id}")
    
    def get_simulation_path(self) -> Optional[Path]:
        """Get the current simulation directory path."""
        return self.simulation_dir 