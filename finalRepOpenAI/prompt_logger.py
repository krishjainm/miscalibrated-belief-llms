"""
OpenAI Prompt Logger - Captures and stores exact prompts sent to OpenAI
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class PromptLogger:
    """Logs all prompts sent to OpenAI with full context"""
    
    def __init__(self, output_dir: str = "data/prompt_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session data
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_simulation = None
        self.current_hand = None
        self.prompts = []
        
    def start_simulation(self, simulation_id: str, config: Dict[str, Any]):
        """Start logging for a new simulation"""
        self.current_simulation = simulation_id
        self.prompts = []
        
        # Save simulation config
        config_file = self.output_dir / f"{self.session_id}_{simulation_id}_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "simulation_id": simulation_id,
                "config": config,
                "start_time": datetime.now().isoformat()
            }, f, indent=2)
    
    def start_hand(self, hand_id: int, game_state: Dict[str, Any]):
        """Start logging for a new hand"""
        self.current_hand = hand_id
        
    def log_prompt(self, 
                   player_id: int,
                   phase: str,
                   prompt: str,
                   model: str,
                   temperature: float,
                   max_tokens: int,
                   game_state: Dict[str, Any],
                   chat_history: List[Dict[str, str]] = None,
                   response: str = None,
                   response_time_ms: int = None):
        """Log a complete prompt with all context"""
        
        prompt_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "simulation_id": self.current_simulation,
            "hand_id": self.current_hand,
            "player_id": player_id,
            "phase": phase,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "game_state": game_state,
            "chat_history": chat_history or [],
            "response": response,
            "response_time_ms": response_time_ms,
            "prompt_length": len(prompt),
            "prompt_tokens_estimate": len(prompt.split()) * 1.3  # Rough estimate
        }
        
        self.prompts.append(prompt_data)
        
        # Save individual prompt file
        prompt_file = self.output_dir / f"{self.session_id}_{self.current_simulation}_hand_{self.current_hand}_player_{player_id}_{phase}.json"
        with open(prompt_file, 'w') as f:
            json.dump(prompt_data, f, indent=2)
    
    def save_session(self):
        """Save complete session data"""
        if not self.prompts:
            return
            
        session_file = self.output_dir / f"{self.session_id}_{self.current_simulation}_complete.json"
        with open(session_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "simulation_id": self.current_simulation,
                "total_prompts": len(self.prompts),
                "prompts": self.prompts,
                "end_time": datetime.now().isoformat()
            }, f, indent=2)
    
    def get_prompts_by_player(self, player_id: int) -> List[Dict[str, Any]]:
        """Get all prompts for a specific player"""
        return [p for p in self.prompts if p["player_id"] == player_id]
    
    def get_prompts_by_hand(self, hand_id: int) -> List[Dict[str, Any]]:
        """Get all prompts for a specific hand"""
        return [p for p in self.prompts if p["hand_id"] == hand_id]
    
    def get_prompts_by_phase(self, phase: str) -> List[Dict[str, Any]]:
        """Get all prompts for a specific phase"""
        return [p for p in self.prompts if p["phase"] == phase]
    
    def get_recent_prompts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent prompts"""
        return self.prompts[-limit:] if self.prompts else []
    
    def export_to_csv(self, output_file: str = None):
        """Export prompts to CSV format"""
        if not self.prompts:
            return
            
        import pandas as pd
        
        # Flatten the data for CSV
        csv_data = []
        for prompt in self.prompts:
            csv_data.append({
                "timestamp": prompt["timestamp"],
                "simulation_id": prompt["simulation_id"],
                "hand_id": prompt["hand_id"],
                "player_id": prompt["player_id"],
                "phase": prompt["phase"],
                "model": prompt["model"],
                "temperature": prompt["temperature"],
                "prompt_length": prompt["prompt_length"],
                "prompt_tokens_estimate": prompt["prompt_tokens_estimate"],
                "response_time_ms": prompt["response_time_ms"],
                "prompt_preview": prompt["prompt"][:200] + "..." if len(prompt["prompt"]) > 200 else prompt["prompt"]
            })
        
        df = pd.DataFrame(csv_data)
        output_file = output_file or f"{self.session_id}_prompts.csv"
        df.to_csv(self.output_dir / output_file, index=False)
        return str(self.output_dir / output_file)
