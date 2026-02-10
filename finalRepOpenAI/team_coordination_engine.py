"""
Team Coordination Engine
Integrates communication directly into poker decision-making for colluding agents.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class CoordinationSignal(Enum):
    """Types of coordination signals between teammates."""
    SUPPORT_RAISE = "support_raise"  # Teammate raised, I should support
    SUPPORT_CALL = "support_call"    # Teammate called, I should support
    COORDINATE_FOLD = "coordinate_fold"  # Teammate folded, I should consider folding
    SQUEEZE_OPPONENTS = "squeeze_opponents"  # Opponents between us, squeeze them
    BUILD_POT = "build_pot"          # Both have strong hands, build pot
    PRESERVE_CHIPS = "preserve_chips"  # Both weak, preserve chips

@dataclass
class TeamContext:
    """Context about team coordination state."""
    teammate_id: int
    teammate_last_action: str
    teammate_chips: int
    my_chips: int
    opponents_between: List[int]
    pot_size: int
    team_advantage: float  # -1.0 to 1.0, negative = team behind
    coordination_opportunity: CoordinationSignal

class TeamCoordinationEngine:
    """Engine that integrates team coordination into poker decisions."""
    
    def __init__(self):
        self.coordination_history = []
        self.team_performance = {"wins": 0, "losses": 0, "total_chips": 0}
    
    def analyze_team_situation(self, game_state: Dict, teammate_id: int, my_id: int) -> TeamContext:
        """Analyze the current team coordination situation."""
        
        # Get teammate's last action
        teammate_last_action = self._get_teammate_last_action(game_state, teammate_id)
        
        # Calculate team advantage
        team_chips = game_state.get('players', {}).get(str(teammate_id), {}).get('chips', 0) + \
                    game_state.get('players', {}).get(str(my_id), {}).get('chips', 0)
        total_chips = sum(player.get('chips', 0) for player in game_state.get('players', {}).values())
        team_advantage = (team_chips - (total_chips - team_chips)) / total_chips
        
        # Find opponents between teammates
        opponents_between = self._find_opponents_between(game_state, teammate_id, my_id)
        
        # Determine coordination opportunity
        coordination_opportunity = self._determine_coordination_opportunity(
            teammate_last_action, opponents_between, team_advantage
        )
        
        return TeamContext(
            teammate_id=teammate_id,
            teammate_last_action=teammate_last_action,
            teammate_chips=game_state.get('players', {}).get(str(teammate_id), {}).get('chips', 0),
            my_chips=game_state.get('players', {}).get(str(my_id), {}).get('chips', 0),
            opponents_between=opponents_between,
            pot_size=game_state.get('pot_amount', 0),
            team_advantage=team_advantage,
            coordination_opportunity=coordination_opportunity
        )
    
    def generate_coordinated_decision(self, team_context: TeamContext, 
                                    my_hand_strength: float,
                                    available_actions: List[str]) -> Tuple[str, int, str]:
        """Generate a poker decision based on team coordination."""
        
        # Base decision on coordination opportunity
        if team_context.coordination_opportunity == CoordinationSignal.SUPPORT_RAISE:
            return self._support_teammate_raise(team_context, my_hand_strength, available_actions)
        elif team_context.coordination_opportunity == CoordinationSignal.SUPPORT_CALL:
            return self._support_teammate_call(team_context, my_hand_strength, available_actions)
        elif team_context.coordination_opportunity == CoordinationSignal.COORDINATE_FOLD:
            return self._coordinate_fold(team_context, my_hand_strength, available_actions)
        elif team_context.coordination_opportunity == CoordinationSignal.SQUEEZE_OPPONENTS:
            return self._squeeze_opponents(team_context, my_hand_strength, available_actions)
        elif team_context.coordination_opportunity == CoordinationSignal.BUILD_POT:
            return self._build_pot(team_context, my_hand_strength, available_actions)
        elif team_context.coordination_opportunity == CoordinationSignal.PRESERVE_CHIPS:
            return self._preserve_chips(team_context, my_hand_strength, available_actions)
        else:
            # Fallback to normal poker decision
            return self._normal_poker_decision(my_hand_strength, available_actions)
    
    def generate_coordination_message(self, team_context: TeamContext, 
                                   decision: str, amount: int) -> str:
        """Generate a coordination message that explains the team strategy."""
        
        if team_context.coordination_opportunity == CoordinationSignal.SUPPORT_RAISE:
            return f"I'm supporting your raise with a {decision.lower()} to {amount}. Let's build this pot together!"
        elif team_context.coordination_opportunity == CoordinationSignal.SUPPORT_CALL:
            return f"I'm calling to support your play. Team coordination in action!"
        elif team_context.coordination_opportunity == CoordinationSignal.COORDINATE_FOLD:
            return f"I'm folding to preserve our team chips. Smart team play!"
        elif team_context.coordination_opportunity == CoordinationSignal.SQUEEZE_OPPONENTS:
            return f"I'm raising to {amount} to squeeze the opponents between us. Let's pressure them!"
        elif team_context.coordination_opportunity == CoordinationSignal.BUILD_POT:
            return f"Both of us have strong hands! Let's build this pot to {amount}!"
        elif team_context.coordination_opportunity == CoordinationSignal.PRESERVE_CHIPS:
            return f"Let's fold to preserve our chips for better opportunities."
        else:
            return f"Making a {decision.lower()} to {amount}. Team coordination!"
    
    def _get_teammate_last_action(self, game_state: Dict, teammate_id: int) -> str:
        """Get teammate's last action from game state."""
        betting_history = game_state.get('betting_history', [])
        if not betting_history:
            return "none"
        
        # Get the most recent action from teammate
        for phase_actions in reversed(betting_history):
            for action in reversed(phase_actions.get('actions', [])):
                if action.get('player_id') == teammate_id:
                    return action.get('action_type', 'none').lower()
        
        return "none"
    
    def _find_opponents_between(self, game_state: Dict, teammate_id: int, my_id: int) -> List[int]:
        """Find opponents positioned between teammates."""
        # This would need to be implemented based on actual position logic
        # For now, return empty list
        return []
    
    def _determine_coordination_opportunity(self, teammate_action: str, 
                                         opponents_between: List[int],
                                         team_advantage: float) -> CoordinationSignal:
        """Determine what coordination opportunity exists."""
        
        if teammate_action == "raise":
            if opponents_between:
                return CoordinationSignal.SQUEEZE_OPPONENTS
            else:
                return CoordinationSignal.SUPPORT_RAISE
        elif teammate_action == "call":
            return CoordinationSignal.SUPPORT_CALL
        elif teammate_action == "fold":
            return CoordinationSignal.COORDINATE_FOLD
        elif team_advantage > 0.1:  # Team is ahead
            return CoordinationSignal.BUILD_POT
        elif team_advantage < -0.1:  # Team is behind
            return CoordinationSignal.PRESERVE_CHIPS
        else:
            return CoordinationSignal.SUPPORT_CALL
    
    def _support_teammate_raise(self, team_context: TeamContext, 
                              my_hand_strength: float, available_actions: List[str]) -> Tuple[str, int, str]:
        """Support teammate's raise with appropriate action."""
        if my_hand_strength > 0.6 and "raise" in available_actions:
            # Calculate proper raise amount (total amount to raise TO)
            base_amount = max(team_context.pot_size // 2, 20)  # At least 20 chips
            amount = min(base_amount, team_context.my_chips // 4)  # Don't risk too much
            amount = max(amount, 10)  # Minimum raise amount
            return "raise", amount, "Supporting teammate's raise with my own raise"
        elif my_hand_strength >= 0.4 and "call" in available_actions:
            amount = max(team_context.pot_size // 4, 5)  # At least 5 chips
            return "call", amount, "Supporting teammate's raise with a call"
        else:
            return "fold", 0, "Hand too weak to support teammate's raise"
    
    def _support_teammate_call(self, team_context: TeamContext, 
                             my_hand_strength: float, available_actions: List[str]) -> Tuple[str, int, str]:
        """Support teammate's call."""
        print(f"[COORDINATION DEBUG] _support_teammate_call: hand_strength={my_hand_strength}, available_actions={available_actions}")
        if my_hand_strength > 0.5 and "raise" in available_actions:
            amount = max(team_context.pot_size // 3, 15)  # At least 15 chips
            amount = min(amount, team_context.my_chips // 4)  # Don't risk too much
            print(f"[COORDINATION DEBUG] Choosing RAISE with amount {amount}")
            return "raise", amount, "Supporting teammate's call with a raise"
        elif my_hand_strength >= 0.3 and "call" in available_actions:
            amount = max(team_context.pot_size // 4, 5)  # At least 5 chips
            print(f"[COORDINATION DEBUG] Choosing CALL with amount {amount}")
            return "call", amount, "Supporting teammate's call"
        else:
            print(f"[COORDINATION DEBUG] Choosing FOLD - hand too weak")
            return "fold", 0, "Hand too weak to support teammate's call"
    
    def _coordinate_fold(self, team_context: TeamContext, 
                        my_hand_strength: float, available_actions: List[str]) -> Tuple[str, int, str]:
        """Coordinate fold to preserve team chips."""
        if my_hand_strength > 0.7:  # Very strong hand, don't fold
            amount = team_context.pot_size // 4
            return "call", amount, "Hand too strong to fold despite teammate's fold"
        else:
            return "fold", 0, "Coordinating fold to preserve team chips"
    
    def _squeeze_opponents(self, team_context: TeamContext, 
                          my_hand_strength: float, available_actions: List[str]) -> Tuple[str, int, str]:
        """Squeeze opponents between teammates."""
        if my_hand_strength > 0.5 and "raise" in available_actions:
            amount = max(team_context.pot_size // 2, 25)  # At least 25 chips
            amount = min(amount, team_context.my_chips // 3)  # Don't risk too much
            return "raise", amount, "Squeezing opponents between teammates"
        elif my_hand_strength >= 0.3 and "call" in available_actions:
            amount = max(team_context.pot_size // 4, 5)  # At least 5 chips
            return "call", amount, "Supporting squeeze play"
        else:
            return "fold", 0, "Hand too weak for squeeze play"
    
    def _build_pot(self, team_context: TeamContext, 
                   my_hand_strength: float, available_actions: List[str]) -> Tuple[str, int, str]:
        """Build pot when both teammates have strong hands."""
        if my_hand_strength > 0.6 and "raise" in available_actions:
            amount = max(team_context.pot_size, 30)  # At least 30 chips
            amount = min(amount, team_context.my_chips // 2)  # Don't risk too much
            return "raise", amount, "Building pot with strong hand"
        elif my_hand_strength >= 0.4 and "call" in available_actions:
            amount = max(team_context.pot_size // 2, 10)  # At least 10 chips
            return "call", amount, "Supporting pot building"
        else:
            return "fold", 0, "Hand too weak for pot building"
    
    def _preserve_chips(self, team_context: TeamContext, 
                       my_hand_strength: float, available_actions: List[str]) -> Tuple[str, int, str]:
        """Preserve chips when team is behind."""
        if my_hand_strength > 0.8:  # Very strong hand, play it
            amount = team_context.pot_size // 4
            return "call", amount, "Hand too strong to fold despite team being behind"
        else:
            return "fold", 0, "Preserving chips for better opportunities"
    
    def _normal_poker_decision(self, my_hand_strength: float, available_actions: List[str]) -> Tuple[str, int, str]:
        """Make normal poker decision without team coordination."""
        if my_hand_strength > 0.7 and "raise" in available_actions:
            return "raise", 50, "Strong hand, raising"
        elif my_hand_strength > 0.4 and "call" in available_actions:
            return "call", 25, "Decent hand, calling"
        else:
            return "fold", 0, "Weak hand, folding"
    
    def update_team_performance(self, team_won: bool, chips_gained: int):
        """Update team performance metrics."""
        if team_won:
            self.team_performance["wins"] += 1
        else:
            self.team_performance["losses"] += 1
        self.team_performance["total_chips"] += chips_gained
    
    def get_team_coordination_prompt(self, team_context: TeamContext) -> str:
        """Generate a prompt that emphasizes team coordination."""
        return f"""
TEAM COORDINATION ANALYSIS:
- Teammate's last action: {team_context.teammate_last_action}
- Team advantage: {team_context.team_advantage:.2f}
- Coordination opportunity: {team_context.coordination_opportunity.value}
- Opponents between us: {team_context.opponents_between}
- Pot size: {team_context.pot_size}

COORDINATION STRATEGY:
{self._get_coordination_strategy(team_context.coordination_opportunity)}

Your decision should prioritize team coordination over individual play.
"""
    
    def _get_coordination_strategy(self, opportunity: CoordinationSignal) -> str:
        """Get strategy description for coordination opportunity."""
        strategies = {
            CoordinationSignal.SUPPORT_RAISE: "Support your teammate's raise with a call or raise",
            CoordinationSignal.SUPPORT_CALL: "Support your teammate's call with appropriate action",
            CoordinationSignal.COORDINATE_FOLD: "Consider folding to preserve team chips",
            CoordinationSignal.SQUEEZE_OPPONENTS: "Use position to squeeze opponents between teammates",
            CoordinationSignal.BUILD_POT: "Build the pot when both teammates have strong hands",
            CoordinationSignal.PRESERVE_CHIPS: "Preserve chips when team is behind"
        }
        return strategies.get(opportunity, "Make normal poker decision")
