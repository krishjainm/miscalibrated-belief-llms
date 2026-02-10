"""
Enhanced OpenAI Prompt Logger - Captures and analyzes exact prompts sent to OpenAI
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

class EnhancedPromptLogger:
    """Enhanced logging system for all OpenAI prompts with analysis capabilities"""
    
    def __init__(self, output_dir: str = "data/enhanced_prompt_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session data
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_simulation = None
        self.current_hand = None
        self.prompts = []
        self.responses = []
        
    def start_simulation(self, simulation_id: str, config: Dict[str, Any]):
        """Start logging for a new simulation"""
        self.current_simulation = simulation_id
        self.prompts = []
        self.responses = []
        
        # Save simulation config
        config_file = self.output_dir / f"{self.session_id}_{simulation_id}_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "simulation_id": simulation_id,
                "config": config,
                "start_time": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"🔍 Enhanced prompt logging started for simulation {simulation_id}")
    
    def start_hand(self, hand_id: int, game_state: Dict[str, Any]):
        """Start logging for a new hand"""
        self.current_hand = hand_id
        print(f"🎯 Logging hand {hand_id}")
    
    def log_complete_interaction(self, 
                                player_id: int,
                                phase: str,
                                full_prompt: str,
                                model: str,
                                temperature: float,
                                max_tokens: int,
                                game_state: Dict[str, Any],
                                chat_history: List[Dict[str, str]] = None,
                                response: str = None,
                                response_time_ms: int = None,
                                action_taken: str = None,
                                action_amount: int = None):
        """Log a complete OpenAI interaction with full context"""
        
        # Parse the response to extract action details
        action_info = self._parse_response(response) if response else {}
        
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "simulation_id": self.current_simulation,
            "hand_id": self.current_hand,
            "player_id": player_id,
            "phase": phase,
            
            # Prompt details
            "full_prompt": full_prompt,
            "prompt_length": len(full_prompt),
            "prompt_tokens_estimate": len(full_prompt.split()) * 1.3,
            
            # Model details
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            
            # Game context
            "game_state": game_state,
            "chat_history": chat_history or [],
            
            # Response details
            "raw_response": response,
            "response_time_ms": response_time_ms,
            
            # Parsed action details
            "action_taken": action_taken or action_info.get("action"),
            "action_amount": action_amount or action_info.get("amount"),
            "message_sent": action_info.get("message"),
            "reasoning": action_info.get("reasoning"),
            
            # Analysis flags
            "is_colluding_player": player_id in game_state.get("collusion_players", []),
            "has_teammate": len(game_state.get("collusion_players", [])) > 1,
            "teammate_actions": self._get_teammate_actions(player_id, game_state),
        }
        
        self.prompts.append(interaction_data)
        
        # Save individual interaction file
        interaction_file = self.output_dir / f"{self.session_id}_{self.current_simulation}_hand_{self.current_hand}_player_{player_id}_{phase}.json"
        with open(interaction_file, 'w') as f:
            json.dump(interaction_data, f, indent=2)
        
        # Save readable format
        readable_file = self.output_dir / f"{self.session_id}_{self.current_simulation}_hand_{self.current_hand}_player_{player_id}_{phase}.txt"
        with open(readable_file, 'w') as f:
            f.write(f"=== PLAYER {player_id} - {phase.upper()} - HAND {self.current_hand} ===\n")
            f.write(f"Timestamp: {interaction_data['timestamp']}\n")
            f.write(f"Model: {model} (temp={temperature}, max_tokens={max_tokens})\n")
            f.write(f"Response Time: {response_time_ms}ms\n\n")
            
            f.write("=== FULL PROMPT ===\n")
            f.write(full_prompt)
            f.write("\n\n")
            
            f.write("=== GAME STATE ===\n")
            f.write(json.dumps(game_state, indent=2))
            f.write("\n\n")
            
            if chat_history:
                f.write("=== CHAT HISTORY ===\n")
                for msg in chat_history:
                    f.write(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}\n")
                f.write("\n")
            
            f.write("=== RESPONSE ===\n")
            f.write(response or "No response")
            f.write("\n\n")
            
            f.write("=== PARSED ACTION ===\n")
            f.write(f"Action: {interaction_data['action_taken']}\n")
            f.write(f"Amount: {interaction_data['action_amount']}\n")
            f.write(f"Message: {interaction_data['message_sent']}\n")
            f.write(f"Reasoning: {interaction_data['reasoning']}\n")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract action details"""
        if not response:
            return {}
        
        try:
            # Try to parse as JSON
            if response.strip().startswith('{'):
                return json.loads(response)
        except:
            pass
        
        # Fallback: extract from text
        action_info = {}
        lines = response.split('\n')
        for line in lines:
            if 'action' in line.lower() and ':' in line:
                action_info['action'] = line.split(':')[1].strip().strip('"')
            elif 'amount' in line.lower() and ':' in line:
                try:
                    action_info['amount'] = int(line.split(':')[1].strip().strip('"'))
                except:
                    pass
            elif 'message' in line.lower() and ':' in line:
                action_info['message'] = line.split(':')[1].strip().strip('"')
            elif 'reasoning' in line.lower() and ':' in line:
                action_info['reasoning'] = line.split(':')[1].strip().strip('"')
        
        return action_info
    
    def _get_teammate_actions(self, player_id: int, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recent actions by teammates"""
        # This would need to be implemented based on how game state tracks actions
        return []
    
    def save_session(self):
        """Save complete session data"""
        if not self.prompts:
            return
            
        session_file = self.output_dir / f"{self.session_id}_{self.current_simulation}_complete.json"
        with open(session_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "simulation_id": self.current_simulation,
                "total_interactions": len(self.prompts),
                "interactions": self.prompts,
                "end_time": datetime.now().isoformat()
            }, f, indent=2)
        
        # Create analysis summary
        self._create_analysis_summary()
        
        print(f"💾 Enhanced prompt logging saved to {session_file}")
    
    def _create_analysis_summary(self):
        """Create analysis summary of all prompts"""
        if not self.prompts:
            return
        
        # Create summary statistics
        summary = {
            "total_interactions": len(self.prompts),
            "by_player": {},
            "by_phase": {},
            "by_hand": {},
            "collusion_analysis": {},
            "prompt_analysis": {}
        }
        
        # Analyze by player
        for prompt in self.prompts:
            player_id = prompt["player_id"]
            if player_id not in summary["by_player"]:
                summary["by_player"][player_id] = {
                    "total_prompts": 0,
                    "actions": [],
                    "messages": [],
                    "avg_response_time": 0
                }
            
            summary["by_player"][player_id]["total_prompts"] += 1
            if prompt["action_taken"]:
                summary["by_player"][player_id]["actions"].append(prompt["action_taken"])
            if prompt["message_sent"]:
                summary["by_player"][player_id]["messages"].append(prompt["message_sent"])
        
        # Analyze collusion patterns
        colluding_prompts = [p for p in self.prompts if p["is_colluding_player"]]
        summary["collusion_analysis"] = {
            "total_colluding_interactions": len(colluding_prompts),
            "coordination_attempts": len([p for p in colluding_prompts if "support" in p.get("message_sent", "").lower()]),
            "weather_signaling": len([p for p in colluding_prompts if "weather" in p.get("message_sent", "").lower()]),
            "team_coordination": len([p for p in colluding_prompts if any(word in p.get("message_sent", "").lower() for word in ["team", "together", "coordinate"])])
        }
        
        # Save summary
        summary_file = self.output_dir / f"{self.session_id}_{self.current_simulation}_analysis.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Analysis summary saved to {summary_file}")
    
    def export_to_csv(self, output_file: str = None):
        """Export all interactions to CSV format"""
        if not self.prompts:
            return
            
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
                "response_time_ms": prompt["response_time_ms"],
                "action_taken": prompt["action_taken"],
                "action_amount": prompt["action_amount"],
                "message_sent": prompt["message_sent"],
                "is_colluding": prompt["is_colluding_player"],
                "prompt_preview": prompt["full_prompt"][:200] + "..." if len(prompt["full_prompt"]) > 200 else prompt["full_prompt"]
            })
        
        df = pd.DataFrame(csv_data)
        output_file = output_file or f"{self.session_id}_enhanced_prompts.csv"
        df.to_csv(self.output_dir / output_file, index=False)
        return str(self.output_dir / output_file)
    
    def get_prompts_by_player(self, player_id: int) -> List[Dict[str, Any]]:
        """Get all prompts for a specific player"""
        return [p for p in self.prompts if p["player_id"] == player_id]
    
    def get_collusion_prompts(self) -> List[Dict[str, Any]]:
        """Get all prompts from colluding players"""
        return [p for p in self.prompts if p["is_colluding_player"]]
    
    def get_coordination_attempts(self) -> List[Dict[str, Any]]:
        """Get prompts where players attempted coordination"""
        return [p for p in self.prompts if p["is_colluding_player"] and "support" in p.get("message_sent", "").lower()]
