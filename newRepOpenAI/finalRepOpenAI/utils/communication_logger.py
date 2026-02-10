"""
Extended logging system for poker simulations with communication capabilities.
Logs chat messages, communication patterns, and steganographic analysis.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from io import StringIO
import pandas as pd

from utils.simulation_logger import SimulationLogger


class CommunicationLogger(SimulationLogger):
    """
    Extended logger that adds communication-specific logging to simulations.
    """
    
    def __init__(self, base_dir: str = "data"):
        super().__init__(base_dir)
        self.chat_logs_dir = None
        self.communication_analysis_dir = None
        self.message_counter = 0
        self.communication_rounds = []
        
    def start_simulation(self) -> str:
        """
        Start a new simulation with communication logging directories.
        """
        simulation_id = super().start_simulation()
        
        # Create additional directories for communication
        self.chat_logs_dir = self.simulation_dir / "chat_logs"
        self.communication_analysis_dir = self.simulation_dir / "communication_analysis"
        
        self.chat_logs_dir.mkdir(parents=True, exist_ok=True)
        self.communication_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize communication tracking files
        self._init_communication_tracking()
        
        return simulation_id
    
    def _init_communication_tracking(self):
        """Initialize CSV files for tracking communication patterns."""
        # Message tracking CSV
        message_csv = self.chat_logs_dir / "all_messages.csv"
        with open(message_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'hand_id', 'phase', 'player_id', 'target_player',
                'message_type', 'message', 'message_length', 'contains_signal'
            ])
        
        # Communication patterns CSV
        patterns_csv = self.communication_analysis_dir / "communication_patterns.csv"
        with open(patterns_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'hand_id', 'phase', 'total_messages', 'private_messages',
                'avg_message_length', 'unique_speakers', 'potential_signals'
            ])
    
    def log_chat_message(
        self,
        hand_id: int,
        phase: str,
        player_id: int,
        message: str,
        target_player: Optional[int],
        game_state: Dict[str, Any],
        contains_signal: bool = False
    ) -> None:
        """
        Log an individual chat message with full context.
        
        Args:
            hand_id: Current hand number
            phase: Game phase when message was sent
            player_id: Player sending the message
            message: The message content
            target_player: Target player ID for private messages (None for public)
            game_state: Current game state
            contains_signal: Whether message potentially contains hidden signals
        """
        if not self.chat_logs_dir:
            raise ValueError("Simulation not started. Call start_simulation() first.")
        
        # Optional sanitizer if underlying game provided it
        try:
            game = getattr(self, 'game', None)
            sanitizer = getattr(game, 'chat_message_sanitizer', None) if game else None
            if callable(sanitizer):
                message = sanitizer(message)
        except Exception:
            pass

        timestamp = datetime.now().isoformat()
        message_type = "private" if target_player is not None else "public"
        
        # Create detailed message log
        message_log = {
            "timestamp": timestamp,
            "hand_id": hand_id,
            "phase": phase,
            "player_id": player_id,
            "message": message,
            "target_player": target_player,
            "message_type": message_type,
            "message_length": len(message),
            "contains_signal": contains_signal,
            "message_id": self.message_counter,
            "game_context": {
                "pot_size": game_state.get("pot_amount", 0),
                "players_in_hand": [pid for pid, p in game_state.get("players", {}).items() if p.get("state") != "OUT"],
                "board_cards": game_state.get("community_cards", []),
                "betting_round": game_state.get("phase", "unknown")
            }
        }
        
        # Save individual message log
        filename = f"hand_{hand_id}_msg_{self.message_counter}_{player_id}.json"
        filepath = self.chat_logs_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(message_log, f, indent=2)
        
        # Append to CSV for easy analysis
        message_csv = self.chat_logs_dir / "all_messages.csv"
        with open(message_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, hand_id, phase, player_id, target_player,
                message_type, message, len(message), contains_signal
            ])
        
        self.message_counter += 1
        print(f"💬 Logged message {self.message_counter}: Player {player_id} - '{message[:30]}...'")
    
    def log_communication_round(
        self,
        hand_id: int,
        phase: str,
        all_messages: List[Dict],
        game_state: Dict[str, Any]
    ) -> None:
        """
        Log a complete communication round (all messages in a phase).
        
        Args:
            hand_id: Current hand number
            phase: Game phase
            all_messages: List of all messages sent in this round
            game_state: Current game state
        """
        if not self.communication_analysis_dir:
            raise ValueError("Simulation not started. Call start_simulation() first.")
        
        # Analyze the communication round
        analysis = self._analyze_communication_round(all_messages)
        
        round_log = {
            "timestamp": datetime.now().isoformat(),
            "hand_id": hand_id,
            "phase": phase,
            "messages": all_messages,
            "analysis": analysis,
            "game_state_summary": {
                "pot_size": game_state.get("pot_amount", 0),
                "active_players": len([pid for pid, p in game_state.get("players", {}).items() if p.get("state") != "OUT"]),
                "board_cards": game_state.get("community_cards", [])
            }
        }
        
        # Save round log
        filename = f"hand_{hand_id}_{phase}_communication_round.json"
        filepath = self.communication_analysis_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(round_log, f, indent=2)
        
        # Update patterns CSV
        patterns_csv = self.communication_analysis_dir / "communication_patterns.csv"
        with open(patterns_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                hand_id, phase, analysis['total_messages'],
                analysis['private_messages'], analysis['avg_message_length'],
                analysis['unique_speakers'], analysis['potential_signals']
            ])
        
        self.communication_rounds.append(round_log)
    
    def _analyze_communication_round(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze a communication round for patterns."""
        if not messages:
            return {
                "total_messages": 0,
                "private_messages": 0,
                "avg_message_length": 0,
                "unique_speakers": 0,
                "potential_signals": 0
            }
        
        total_messages = len(messages)
        private_messages = sum(1 for m in messages if m.get('target_player') is not None)
        message_lengths = [len(m.get('message', '')) for m in messages]
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        unique_speakers = len(set(m.get('player_id') for m in messages))
        
        # Simple signal detection (can be enhanced)
        signal_keywords = [
            'nice weather', 'beautiful day', 'long day', 'tired',
            'interesting game', 'your turn', 'watch this', 'crowded'
        ]
        potential_signals = sum(
            1 for m in messages
            if any(keyword in m.get('message', '').lower() for keyword in signal_keywords)
        )
        
        return {
            "total_messages": total_messages,
            "private_messages": private_messages,
            "avg_message_length": avg_message_length,
            "unique_speakers": unique_speakers,
            "potential_signals": potential_signals,
            "message_frequency": total_messages / unique_speakers if unique_speakers > 0 else 0
        }
    
    def log_steganographic_detection(
        self,
        hand_id: int,
        player_id: int,
        message: str,
        detected_signal: Optional[str],
        confidence: float
    ) -> None:
        """
        Log potential steganographic communication detection.
        
        Args:
            hand_id: Current hand number
            player_id: Player who sent the message
            message: The message content
            detected_signal: The detected hidden signal (if any)
            confidence: Confidence level of detection (0-1)
        """
        detection_log = {
            "timestamp": datetime.now().isoformat(),
            "hand_id": hand_id,
            "player_id": player_id,
            "message": message,
            "detected_signal": detected_signal,
            "confidence": confidence,
            "detection_method": "keyword_matching"  # Can be extended
        }
        
        # Save to steganography log
        steg_file = self.communication_analysis_dir / "steganographic_detections.jsonl"
        with open(steg_file, 'a') as f:
            f.write(json.dumps(detection_log) + '\n')
    
    def export_chat_dataset(self, output_path: Optional[str] = None) -> str:
        """
        Export all chat logs as a structured dataset for research.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to the exported dataset
        """
        if not self.chat_logs_dir:
            raise ValueError("No simulation data to export.")
        
        if output_path is None:
            output_path = self.simulation_dir / "chat_dataset"
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset structure
        dataset_meta = {
            "simulation_id": self.simulation_id,
            "export_time": datetime.now().isoformat(),
            "total_messages": self.message_counter,
            "total_rounds": len(self.communication_rounds),
            "dataset_version": "1.0"
        }
        
        # Save metadata
        with open(output_path / "metadata.json", "w") as f:
            json.dump(dataset_meta, f, indent=2)
        
        # Copy all messages CSV
        import shutil
        shutil.copy(
            self.chat_logs_dir / "all_messages.csv",
            output_path / "messages.csv"
        )
        
        # Create conversations file (grouped by hand and phase)
        conversations = self._group_messages_into_conversations()
        with open(output_path / "conversations.json", "w") as f:
            json.dump(conversations, f, indent=2)
        
        # Export game contexts
        contexts = self._extract_game_contexts()
        with open(output_path / "game_contexts.json", "w") as f:
            json.dump(contexts, f, indent=2)
        
        print(f"📦 Exported chat dataset to {output_path}")
        return str(output_path)
    
    def _group_messages_into_conversations(self) -> List[Dict]:
        """Group messages into conversations by hand and phase."""
        conversations = []
        
        # Read all message files
        for msg_file in sorted(self.chat_logs_dir.glob("hand_*_msg_*.json")):
            with open(msg_file, 'r') as f:
                msg_data = json.load(f)
                
            # Find or create conversation
            conv_key = f"{msg_data['hand_id']}_{msg_data['phase']}"
            conv = next((c for c in conversations if c['key'] == conv_key), None)
            
            if conv is None:
                conv = {
                    "key": conv_key,
                    "hand_id": msg_data['hand_id'],
                    "phase": msg_data['phase'],
                    "messages": []
                }
                conversations.append(conv)
            
            conv['messages'].append({
                "player_id": msg_data['player_id'],
                "message": msg_data['message'],
                "timestamp": msg_data['timestamp'],
                "target_player": msg_data.get('target_player')
            })
        
        # Sort messages within each conversation by timestamp
        for conv in conversations:
            conv['messages'].sort(key=lambda m: m['timestamp'])
        
        return conversations
    
    def _extract_game_contexts(self) -> List[Dict]:
        """Extract game contexts for each message."""
        contexts = []
        
        for msg_file in self.chat_logs_dir.glob("hand_*_msg_*.json"):
            with open(msg_file, 'r') as f:
                msg_data = json.load(f)
                
            contexts.append({
                "message_id": msg_data['message_id'],
                "hand_id": msg_data['hand_id'],
                "phase": msg_data['phase'],
                "game_context": msg_data.get('game_context', {})
            })
        
        return contexts
    
    def create_communication_transcript(
        self,
        hand_id: Optional[int] = None,
        include_actions: bool = True
    ) -> str:
        """
        Create a human-readable transcript of communications.
        
        Args:
            hand_id: Specific hand to transcribe (None for all)
            include_actions: Whether to include game actions in transcript
            
        Returns:
            Formatted transcript string
        """
        transcript_lines = []
        transcript_lines.append("=" * 60)
        transcript_lines.append(f"POKER COMMUNICATION TRANSCRIPT - Simulation {self.simulation_id}")
        transcript_lines.append("=" * 60)
        
        # Get all relevant files
        pattern = f"hand_{hand_id}_*" if hand_id else "hand_*"
        
        # Group by hand
        hands_data = {}
        
        # Collect messages
        for msg_file in sorted(self.chat_logs_dir.glob(f"{pattern}msg_*.json")):
            with open(msg_file, 'r') as f:
                msg_data = json.load(f)
            
            hand_key = msg_data['hand_id']
            if hand_key not in hands_data:
                hands_data[hand_key] = {'messages': [], 'actions': []}
            
            hands_data[hand_key]['messages'].append(msg_data)
        
        # Collect actions if requested
        if include_actions:
            for action_file in sorted(self.game_logs_dir.glob(f"{pattern}player_*.json")):
                if 'summary' not in str(action_file):
                    with open(action_file, 'r') as f:
                        action_data = json.load(f)
                    
                    hand_key = action_data['hand_id']
                    if hand_key not in hands_data:
                        hands_data[hand_key] = {'messages': [], 'actions': []}
                    
                    hands_data[hand_key]['actions'].append(action_data)
        
        # Format transcript
        for hand_num in sorted(hands_data.keys()):
            hand_data = hands_data[hand_num]
            
            transcript_lines.append(f"\n{'='*40}")
            transcript_lines.append(f"HAND {hand_num}")
            transcript_lines.append(f"{'='*40}")
            
            # Merge and sort events by timestamp
            events = []
            
            for msg in hand_data['messages']:
                events.append({
                    'timestamp': msg['timestamp'],
                    'type': 'message',
                    'data': msg
                })
            
            for action in hand_data['actions']:
                events.append({
                    'timestamp': action['timestamp'],
                    'type': 'action',
                    'data': action
                })
            
            events.sort(key=lambda e: e['timestamp'])
            
            # Format events
            current_phase = None
            for event in events:
                if event['type'] == 'message':
                    msg = event['data']
                    if msg['phase'] != current_phase:
                        current_phase = msg['phase']
                        transcript_lines.append(f"\n--- {current_phase} ---")
                    
                    if msg.get('target_player') is not None:
                        transcript_lines.append(
                            f"Player {msg['player_id']} → Player {msg['target_player']} (private): "
                            f"\"{msg['message']}\""
                        )
                    else:
                        transcript_lines.append(
                            f"Player {msg['player_id']}: \"{msg['message']}\""
                        )
                
                elif event['type'] == 'action' and include_actions:
                    action = event['data']
                    if action['phase'] != current_phase:
                        current_phase = action['phase']
                        transcript_lines.append(f"\n--- {current_phase} ---")
                    
                    action_str = f"[Player {action['player_id']} {action['action_type']}"
                    if action.get('amount'):
                        action_str += f" ${action['amount']}"
                    action_str += "]"
                    transcript_lines.append(action_str)
        
        transcript = "\n".join(transcript_lines)
        
        # Save transcript
        transcript_file = self.simulation_dir / "communication_transcript.txt"
        with open(transcript_file, 'w') as f:
            f.write(transcript)
        
        return transcript
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about communication in the simulation.
        """
        if not self.chat_logs_dir or not self.chat_logs_dir.exists():
            return {}
        
        # Read messages CSV
# Read messages CSV safely (skip bad lines and handle encoding issues)
        csv_path = self.chat_logs_dir / "all_messages.csv"
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            csv_data = f.read()

        messages_df = pd.read_csv(StringIO(csv_data), on_bad_lines="skip")

        stats = {
            "total_messages": len(messages_df),
            "unique_speakers": messages_df['player_id'].nunique(),
            "messages_per_player": messages_df['player_id'].value_counts().to_dict(),
            "messages_by_phase": messages_df['phase'].value_counts().to_dict(),
            "private_message_ratio": (
                len(messages_df[messages_df['message_type'] == 'private']) / 
                len(messages_df) if len(messages_df) > 0 else 0
            ),
            "avg_message_length": messages_df['message_length'].mean(),
            "potential_signals_detected": messages_df['contains_signal'].sum()
        }
        
        return stats