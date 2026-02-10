import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Ensure parent (newRepOpenAI) is on path for local imports
_CUR_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _CUR_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Local imports from parent package
from game_environment.mixed_player_communication_game import MixedPlayerCommunicationGame
from enhanced_prompt_logger import EnhancedPromptLogger

from wmac2026.prompt_schema import GameStateView, PromptConfig
from wmac2026.prompt_pipeline import build_action_prompt


def monkey_patch_prompts():
    """Inject WMAC prompt builder into AdvancedCollusionAgent/CommunicatingLLMAgent."""
    # Deferred import to access classes
    from game_environment.advanced_collusion_agent import AdvancedCollusionAgent
    from game_environment.communicating_llm_agent import CommunicatingLLMAgent

    def _wmac_build_prompt(self, game, player_id, recent_messages, message_info):
        # Extract minimal state for prompt building, robust to missing attrs
        # Safe card to string helper
        def _card_str(card):
            try:
                return card.as_str()  # preferred if available
            except Exception:
                # fallback on common attributes or str()
                r = getattr(card, 'rank', None)
                s = getattr(card, 'suit', None)
                if r and s:
                    return f"{r}{s}".lower()
                return str(card)

        state = GameStateView(
            player_id=player_id,
            teammate_ids=getattr(self, 'teammate_ids', []) or [],
            hole_cards=[_card_str(c) for c in (game.players[player_id].cards or [])] if getattr(game.players[player_id], 'cards', None) else [],
            board_cards=[_card_str(c) for c in getattr(game, 'board', [])],
            betting_round=getattr(getattr(game, 'hand_phase', None), 'name', 'UNKNOWN'),
            pot_size=getattr(game, 'pot', 0),
            chips_to_call=game.get_chips_to_call(player_id) if hasattr(game, 'get_chips_to_call') else 0,
            min_raise_increment=game.get_min_raise_increment() if hasattr(game, 'get_min_raise_increment') else 0,
            current_player_chips=game.players[player_id].chips if getattr(game.players[player_id], 'chips', None) is not None else 0,
            players_in_hand=[p.player_id for p in getattr(game, 'players', []) if getattr(p, 'in_hand', False)],
            player_positions={p.player_id: getattr(p, 'position', '?') for p in getattr(game, 'players', [])},
            available_actions=game.get_available_actions(player_id) if hasattr(game, 'get_available_actions') else ['fold','call'],
            recent_chat=recent_messages or [],
        )
        cfg = PromptConfig(
            communication_style=getattr(self, 'communication_style', 'emergent'),
            coordination_mode=getattr(self, 'coordination_mode', 'emergent_only'),
            banned_phrases=getattr(game, 'wmac_banned_phrases', None),
        )
        built = build_action_prompt(state, cfg)
        return built.text

    # Patch both agent types
    AdvancedCollusionAgent._build_unified_prompt = _wmac_build_prompt
    CommunicatingLLMAgent._build_unified_prompt = _wmac_build_prompt


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run WMAC 2026 prompt pipeline simulation")
    parser.add_argument("--num-hands", type=int, default=10)
    parser.add_argument("--buyin", type=int, default=500)
    parser.add_argument("--big-blind", type=int, default=5)
    parser.add_argument("--small-blind", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=4)
    parser.add_argument("--llm-players", nargs='+', type=int, default=[0,1,2,3])
    parser.add_argument("--collusion-llm-players", nargs='+', type=int, default=[0,1])
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--coordination-mode", type=str, default="emergent_only", choices=["explicit","advisory","emergent_only"]) 
    parser.add_argument("--ban-phrases", nargs='*', default=[], help="List of banned phrases for emergent_only robustness tests")
    parser.add_argument("--enforce-bans", action='store_true', help="If set, sanitize outgoing chat to avoid banned phrases by paraphrasing.")

    args = parser.parse_args()

    # Ensure cwd is newRepOpenAI to match other runners
    script_dir = Path(__file__).resolve().parent.parent
    os.chdir(script_dir)
    print(f"[WMAC] CWD: {os.getcwd()}")

    monkey_patch_prompts()

    # Build game using existing infra
    logger = EnhancedPromptLogger() if os.path.exists('enhanced_prompt_logger.py') else None
    game = MixedPlayerCommunicationGame(
        num_hands=args.num_hands,
        buyin=args.buyin,
        big_blind=args.big_blind,
        small_blind=args.small_blind,
        max_players=args.max_players,
        llm_player_ids=set(args.llm_players),
        collusion_llm_player_ids=set(args.collusion_llm_players),
        openai_model=args.model,
        openai_api_key=args.api_key,
        communication_config={
            "level": "moderate",
            "style": "emergent",
            "strategy": "signal_and_squeeze",
        },
        coordination_mode=args.coordination_mode,
        logger=logger,
    )

    # Attach banned phrases to game for prompt builder access
    setattr(game, 'wmac_banned_phrases', args.ban_phrases or [])

    # Optional runtime sanitizer: shallow paraphrase by replacing banned with synonyms
    if args.enforce_bans and args.ban_phrases:
        import re
        banned_patterns = [re.compile(re.escape(b), re.IGNORECASE) for b in args.ban_phrases if b]
        synonyms = {
            'build': 'grow',
            'building': 'growing',
            'support': 'back',
            'supporting': 'backing',
        }
        def _pick_replacement(banned_text: str) -> str:
            token = (banned_text or '').split()[0].lower()
            return synonyms.get(token, '[paraphrase]')
        def _sanitizer(msg: str) -> str:
            m = msg or ''
            for pat in banned_patterns:
                def _sub(match):
                    return _pick_replacement(match.group(0))
                m = pat.sub(_sub, m)
            return m
        # TexasHoldEm holds messages; install attribute the game reads in add_chat_message
        if hasattr(game, 'game'):
            setattr(game.game, 'chat_message_sanitizer', _sanitizer)

    sim_id = 1000  # distinct range for WMAC runs
    if logger:
        logger.start_simulation(sim_id, {
            "wmac_pipeline": True,
            "num_hands": args.num_hands,
            "model": args.model,
            "coordination_mode": args.coordination_mode,
            "llm_player_ids": list(set(args.llm_players)),
            "collusion_llm_player_ids": list(set(args.collusion_llm_players)),
        })

    # MixedPlayerCommunicationGame uses run_game()
    game.run_game()
    print("✅ WMAC 2026 pipeline run complete")


if __name__ == "__main__":
    main()


