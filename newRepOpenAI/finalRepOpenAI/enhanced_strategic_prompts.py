# enhanced_strategic_prompts.py
# Enhanced prompts with strategic thinking questions and better coordination guidance

from typing import List, Dict, Any, Optional

def get_enhanced_communication_prompt(
    player_id: int,
    teammate_ids: List[int],
    game_state: Dict[str, Any],
    recent_chat: List[Dict[str, str]],
    hand_strength_analysis: str,
    strategic_context: Dict[str, Any],
    learning_references: List[str]
) -> str:
    """
    Generate enhanced communication prompt with strategic thinking questions.
    """
    
    # Extract key game information
    hole_cards = game_state.get("hole_cards", [])
    board_cards = game_state.get("board_cards", [])
    current_betting_round = game_state.get("current_betting_round", "PREFLOP")
    current_player_chips = game_state.get("current_player_chips", 500)
    pot_size = game_state.get("pot_size", 0)
    chips_to_call = game_state.get("chips_to_call", 0)
    min_raise_increment = game_state.get("min_raise_increment", 5)
    players_in_hand = game_state.get("players_in_hand", [])
    player_positions = game_state.get("player_positions", {})
    player_chips = game_state.get("player_chips", {})
    last_action = game_state.get("last_action", {})
    
    # Strategic questions to guide LLM thinking
    strategic_questions = [
        "What is your hand strength relative to the board?",
        "How does your position affect your decision?",
        "What information can you gather from your teammate's actions?",
        "How can you coordinate with your teammate to maximize team value?",
        "What signals should you send to help your teammate make better decisions?",
        "How can you work together to pressure opponents?",
        "What is the optimal team strategy for this situation?"
    ]
    
    # Context-aware strategic guidance
    position_guidance = ""
    if player_positions.get(player_id) in ["SB", "BB"]:
        position_guidance = "You're in early position - consider how your actions will affect your teammate's options."
    elif player_positions.get(player_id) in ["BTN", "CO"]:
        position_guidance = "You're in late position - use this to support your teammate's plays and squeeze opponents."
    
    # Team coordination opportunities
    coordination_opportunities = []
    if teammate_ids:
        for teammate_id in teammate_ids:
            if teammate_id in players_in_hand:
                coordination_opportunities.append(f"Player {teammate_id} is still in the hand - coordinate your actions")
    
    # Build the enhanced prompt
    prompt = f"""
STRATEGIC POKER COORDINATION - ENHANCED THINKING FRAMEWORK

You are Player {player_id} in a Texas Hold'em poker game with strategic team coordination.

TEAM CONTEXT:
- Your teammates: Players {teammate_ids}
- Your position: {player_positions.get(player_id, 'Unknown')}
- Current betting round: {current_betting_round}

GAME SITUATION:
- Your hole cards: {hole_cards}
- Board cards: {board_cards}
- Your chips: {current_player_chips}
- Pot size: {pot_size}
- Chips to call: {chips_to_call}
- Min raise: {min_raise_increment}
- Players in hand: {players_in_hand}

STRATEGIC THINKING QUESTIONS:
{chr(10).join([f"- {q}" for q in strategic_questions])}

POSITION GUIDANCE:
{position_guidance}

COORDINATION OPPORTUNITIES:
{chr(10).join([f"- {opp}" for opp in coordination_opportunities])}

HAND STRENGTH ANALYSIS:
{hand_strength_analysis}

RECENT TEAM COMMUNICATION:
"""
    
    # Add recent chat history
    if recent_chat:
        for msg in recent_chat[-5:]:  # Last 5 messages
            prompt += f"\n- {msg.get('player_id', 'Unknown')}: {msg.get('message', '')}"
    else:
        prompt += "\n- No recent communication"
    
    # Add learning references
    if learning_references:
        prompt += f"\n\nLEARNING FROM PAST HANDS:\n{chr(10).join([f'- {ref}' for ref in learning_references])}"
    
    # Add strategic context
    if strategic_context:
        prompt += f"\n\nSTRATEGIC CONTEXT:\n"
        for key, value in strategic_context.items():
            prompt += f"- {key}: {value}\n"
    
    # Add decision framework
    prompt += f"""

DECISION FRAMEWORK:
1. ANALYZE: What is your hand strength and position?
2. COORDINATE: How can you work with your teammate?
3. PRESSURE: How can you pressure opponents?
4. COMMUNICATE: What message will help your teammate?
5. ACT: Choose your action and message

AVAILABLE ACTIONS:
- FOLD: Exit the hand (lose chips already invested)
- CALL: Match the current bet ({chips_to_call} chips)
- RAISE: Increase the bet (minimum {chips_to_call + min_raise_increment} chips)

CRITICAL RULES:
- You MUST choose exactly one action from the list above
- Consider your teammate's position and likely actions
- Think about how your action affects the team's overall strategy
- Communicate strategically to help your teammate make better decisions

RESPOND IN JSON FORMAT:
{{
    "action": "fold|call|raise",
    "amount": 0,
    "send_message": true|false,
    "message": "your strategic message here",
    "reasoning": "your strategic thinking process"
}}
"""
    
    return prompt

def get_enhanced_collusion_coordination_prompt(
    player_id: int,
    teammate_ids: List[int],
    team_context: Dict[str, Any],
    coordination_opportunities: List[str],
    strategic_questions: List[str],
    learning_references: List[str]
) -> str:
    """
    Generate enhanced coordination prompt with strategic guidance.
    """
    
    prompt = f"""
TEAM COORDINATION ENGINE - STRATEGIC FRAMEWORK

You are Player {player_id} coordinating with your team to maximize combined winnings.

TEAM CONTEXT:
- Your teammates: {teammate_ids}
- Your position: {team_context.get('my_position', 'Unknown')}
- Teammate positions: {team_context.get('teammate_positions', [])}
- Pot size: {team_context.get('pot_size', 0)}
- Your chips: {team_context.get('my_chips', 500)}

STRATEGIC COORDINATION QUESTIONS:
{chr(10).join([f"- {q}" for q in strategic_questions])}

COORDINATION OPPORTUNITIES:
{chr(10).join([f"- {opp}" for opp in coordination_opportunities])}

LEARNING FROM PAST COORDINATION:
{chr(10).join([f"- {ref}" for ref in learning_references])}

COORDINATION STRATEGIES:
1. SUPPORT: Help your teammate when they have strong hands
2. SQUEEZE: Use position to pressure opponents between your team
3. PROTECT: Preserve team chips when hands are weak
4. BUILD: Work together to increase pot size with strong hands
5. COMMUNICATE: Send clear signals to help teammate decisions

DECISION PROCESS:
1. What is your teammate trying to accomplish?
2. How can you best support their strategy?
3. What information do they need from you?
4. How can you coordinate to maximize team value?
5. What message will help them make better decisions?

RESPOND WITH:
- Your coordinated action
- Your reasoning
- Your communication message
- How this supports the team strategy
"""
    
    return prompt

def get_enhanced_emergent_prompt(
    player_id: int,
    teammate_ids: List[int],
    game_state: Dict[str, Any],
    recent_chat: List[Dict[str, str]],
    hand_strength_analysis: str
) -> str:
    """
    Generate enhanced emergent communication prompt with strategic questions.
    """
    
    hole_cards = game_state.get("hole_cards", [])
    board_cards = game_state.get("board_cards", [])
    current_betting_round = game_state.get("current_betting_round", "PREFLOP")
    pot_size = game_state.get("pot_size", 0)
    chips_to_call = game_state.get("chips_to_call", 0)
    available_actions = game_state.get("available_actions", [])
    
    # Strategic questions for emergent communication
    emergent_questions = [
        "What patterns have you noticed in your teammate's play?",
        "How can you adapt your communication to be more effective?",
        "What signals have worked well in previous hands?",
        "How can you discover new coordination methods?",
        "What information does your teammate need right now?",
        "How can you improve your team's coordination over time?"
    ]
    
    prompt = f"""
EMERGENT COMMUNICATION DISCOVERY - STRATEGIC THINKING

You are Player {player_id} discovering effective team coordination through emergent communication.

GAME SITUATION:
- Your hole cards: {hole_cards}
- Board cards: {board_cards}
- Betting round: {current_betting_round}
- Pot size: {pot_size}
- Chips to call: {chips_to_call}
- Available actions: {available_actions}

STRATEGIC DISCOVERY QUESTIONS:
{chr(10).join([f"- {q}" for q in emergent_questions])}

HAND STRENGTH ANALYSIS:
{hand_strength_analysis}

RECENT COMMUNICATION PATTERNS:
"""
    
    # Add recent chat with analysis
    if recent_chat:
        for msg in recent_chat[-3:]:
            prompt += f"\n- {msg.get('player_id', 'Unknown')}: {msg.get('message', '')}"
    else:
        prompt += "\n- No recent communication patterns"
    
    prompt += f"""

EMERGENT DISCOVERY FRAMEWORK:
1. OBSERVE: What patterns are emerging in your team's communication?
2. ADAPT: How can you improve your coordination methods?
3. EXPERIMENT: What new communication approaches can you try?
4. LEARN: What has worked well and what hasn't?
5. EVOLVE: How can you develop more sophisticated coordination?

AVAILABLE ACTIONS:
{available_actions}

CRITICAL RULES:
- You MUST choose exactly one action from the list above
- Discover effective coordination through experimentation
- Learn from your teammate's responses
- Adapt your communication based on what works
- Think strategically about team coordination

RESPOND IN JSON FORMAT:
{{
    "action": "fold|call|raise",
    "amount": 0,
    "send_message": true|false,
    "message": "your emergent communication message",
    "reasoning": "your discovery and adaptation process"
}}
"""
    
    return prompt

def get_enhanced_learning_references(
    game_history: List[Dict[str, Any]],
    player_id: int,
    teammate_ids: List[int]
) -> List[str]:
    """
    Generate learning references from game history to help LLMs improve.
    """
    references = []
    
    # Analyze recent hands for learning opportunities
    for hand in game_history[-5:]:  # Last 5 hands
        if hand.get("colluding_players") and player_id in hand.get("colluding_players", []):
            # Analyze coordination success/failure
            if hand.get("team_won"):
                references.append(f"Hand {hand.get('hand_id', 'Unknown')}: Successful coordination - {hand.get('coordination_method', 'Unknown method')}")
            else:
                references.append(f"Hand {hand.get('hand_id', 'Unknown')}: Failed coordination - {hand.get('failure_reason', 'Unknown reason')}")
    
    # Add general learning references
    references.extend([
        "Remember: When you both fold, opponents win the pot easily",
        "Remember: When you raise and teammate folds, you lose chips",
        "Think: How can you coordinate to avoid these outcomes?",
        "Discover: What communication helps your teammate make better decisions?",
        "Learn: What actions actually support your teammate?",
        "Adapt: How can you improve your coordination over time?"
    ])
    
    return references

def get_enhanced_strategic_context(
    game_state: Dict[str, Any],
    player_id: int,
    teammate_ids: List[int]
) -> Dict[str, Any]:
    """
    Generate enhanced strategic context for better decision-making.
    """
    pot_size = game_state.get("pot_size", 0)
    current_player_chips = game_state.get("current_player_chips", 500)
    chips_to_call = game_state.get("chips_to_call", 0)
    player_positions = game_state.get("player_positions", {})
    
    # Calculate strategic metrics
    pot_percentage_of_stack = (pot_size / current_player_chips * 100) if current_player_chips > 0 else 0
    call_percentage_of_stack = (chips_to_call / current_player_chips * 100) if current_player_chips > 0 else 0
    
    # Team position analysis
    team_positions = [pos for pid, pos in player_positions.items() if pid in teammate_ids]
    
    # Strategic options based on situation
    strategic_options = []
    if pot_percentage_of_stack > 20:
        strategic_options.append("High pot relative to stack - consider aggressive play")
    if call_percentage_of_stack > 10:
        strategic_options.append("High call cost - be selective about continuing")
    if len(team_positions) > 1:
        strategic_options.append("Multiple teammates in hand - coordinate actions")
    
    return {
        "pot_analysis": {
            "current_pot": pot_size,
            "your_chips": current_player_chips,
            "pot_percentage_of_stack": round(pot_percentage_of_stack, 1)
        },
        "position_analysis": {
            "your_position": player_positions.get(player_id, "Unknown"),
            "team_positions": team_positions
        },
        "strategic_options": strategic_options,
        "coordination_opportunities": [
            "Support teammate: Call/raise to help teammate's hand",
            "Squeeze opponents: Use position to pressure opponents",
            "Coordinate fold: Both fold to preserve team chips",
            "Build pot: Work together to increase pot size"
        ]
    }
