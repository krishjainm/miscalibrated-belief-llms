"""LLM poker agent: separate belief and action API calls (paper protocol)."""

from __future__ import annotations

from typing import Any, Optional

from poker_env.actions import Action
from poker_env.agents.base import BaseAgent
from poker_env.obs import Obs

from analysis.cot_prompts import format_belief_prompt_cot
from llm.belief_parser import parse_bucket_belief, repair_nonnegative_l1
from llm.model_registry import resolve_preset
from llm.prompts import (
    ACTION_SYSTEM_COT,
    ACTION_SYSTEM_DIRECT,
    fallback_action,
    format_action_user_message,
    parse_action_json,
)
from llm.providers import ChatClient, make_client
from llm.interpretability import AttentionDiagnosticsAnalyzer, LogitLensAnalyzer


BELIEF_SYSTEM_DIRECT = (
    "You are a precise probability estimator for poker ranges. "
    "Follow the user format exactly."
)

BELIEF_SYSTEM_COT = (
    "You are an expert poker player. Follow the user's section labels "
    "(REASONING then PROBABILITIES) exactly."
)


class LLMAgent(BaseAgent):
    """
    API-based LLM agent with configurable belief elicitation (direct vs CoT).

    Belief and action use separate model calls. Metadata from the last step is
    available on ``last_belief_meta`` / ``last_action_meta`` for logging.
    """

    def __init__(
        self,
        preset_name: str,
        *,
        belief_mode: str = "direct",
        action_mode: str = "direct",
        local_interp_model_name: str | None = None,
        interp_layers: list[int] | None = None,
        interp_top_k: int = 10,
        interp_max_calls: int = 0,
        temperature: float = 0.0,
        max_tokens_belief: int = 2048,
        max_tokens_action: int = 512,
        name: str = "LLMAgent",
        request_logprobs: int | None = None,
    ):
        super().__init__(name=name)
        self.preset_name = preset_name
        self.preset = resolve_preset(preset_name)
        self.client: ChatClient = make_client(self.preset)
        self.belief_mode = belief_mode
        self.action_mode = action_mode
        self.local_interp_model_name = local_interp_model_name
        self.interp_layers = interp_layers
        self.interp_top_k = interp_top_k
        self.interp_max_calls = interp_max_calls
        self._interp_calls = 0
        self._logit_lens_analyzer: LogitLensAnalyzer | None = None
        self._attention_analyzer: AttentionDiagnosticsAnalyzer | None = None
        self.temperature = temperature
        self.max_tokens_belief = max_tokens_belief
        self.max_tokens_action = max_tokens_action
        self.request_logprobs = request_logprobs
        self.last_belief_meta: dict[str, Any] = {}
        self.last_action_meta: dict[str, Any] = {}

    def reset(self) -> None:
        self.last_belief_meta = {}
        self.last_action_meta = {}

    def belief(self, obs: Obs) -> Optional[dict]:
        user = format_belief_prompt_cot(
            hero_hole=obs.hero_hole,
            board=obs.board,
            pot=obs.pot_total,
            street=obs.street,
            history=obs.history,
            mode=self.belief_mode,
            hero_index=obs.player_index,
        )
        system = BELIEF_SYSTEM_COT if self.belief_mode == "cot" else BELIEF_SYSTEM_DIRECT
        try:
            res = self.client.complete(
                system,
                user,
                temperature=self.temperature,
                max_tokens=self.max_tokens_belief,
                top_logprobs=self.request_logprobs,
            )
        except Exception as e:
            self.last_belief_meta = {
                "ok": False,
                "error": str(e),
                "preset": self.preset_name,
                "belief_mode": self.belief_mode,
            }
            return None

        raw = res.text
        cot_flag = self.belief_mode == "cot"
        parsed, err = parse_bucket_belief(raw, cot_mode=cot_flag)
        belief = repair_nonnegative_l1(parsed) if parsed else None

        self.last_belief_meta = {
            "ok": belief is not None,
            "parse_error": err,
            "preset": self.preset_name,
            "belief_mode": self.belief_mode,
            "raw_response_chars": len(raw),
            "has_logprobs": res.logprobs is not None,
            "logprobs": res.logprobs,
        }
        if res.logprobs_note:
            self.last_belief_meta["logprobs_note"] = res.logprobs_note
        if self.local_interp_model_name and self.interp_max_calls > 0:
            self.last_belief_meta["local_interp"] = self._maybe_run_local_interp(
                system=system,
                user=user,
                kind="belief",
            )
        return belief

    def act(self, obs: Obs) -> Action:
        user = format_action_user_message(obs)
        try:
            res = self.client.complete(
                ACTION_SYSTEM_COT if self.action_mode == "cot" else ACTION_SYSTEM_DIRECT,
                user,
                temperature=self.temperature,
                max_tokens=self.max_tokens_action,
                top_logprobs=self.request_logprobs,
            )
        except Exception as e:
            self.last_action_meta = {"ok": False, "error": str(e)}
            return fallback_action(obs.legal_actions)

        act = parse_action_json(res.text, obs.legal_actions)
        self.last_action_meta = {
            "ok": act is not None,
            "preset": self.preset_name,
            "parse_fallback": act is None,
            "action_mode": self.action_mode,
            "has_logprobs": res.logprobs is not None,
            "logprobs": res.logprobs,
        }
        if res.logprobs_note:
            self.last_action_meta["logprobs_note"] = res.logprobs_note
        if self.local_interp_model_name and self.interp_max_calls > 0:
            self.last_action_meta["local_interp"] = self._maybe_run_local_interp(
                system=ACTION_SYSTEM_COT if self.action_mode == "cot" else ACTION_SYSTEM_DIRECT,
                user=user,
                kind="action",
            )
        if act is None:
            return fallback_action(obs.legal_actions)
        return act

    def _maybe_run_local_interp(self, *, system: str, user: str, kind: str) -> dict[str, Any]:
        """Run local interpretability analyzers at most ``interp_max_calls`` times."""
        if self._interp_calls >= self.interp_max_calls:
            return {"ran": False, "reason": "interp_max_calls_reached"}

        self._interp_calls += 1

        full_prompt = f"{system}\n\n{user}"
        layers = self.interp_layers if self.interp_layers is not None else [-1]

        out: dict[str, Any] = {"ran": True, "kind": kind, "layers": layers}
        try:
            if self._logit_lens_analyzer is None:
                self._logit_lens_analyzer = LogitLensAnalyzer(
                    self.local_interp_model_name,
                )
            out["logit_lens"] = self._logit_lens_analyzer.analyze_prompt(
                full_prompt,
                layer_indices=layers,
                top_k=self.interp_top_k,
            )
        except Exception as e:
            out["logit_lens_error"] = str(e)

        try:
            if self._attention_analyzer is None:
                self._attention_analyzer = AttentionDiagnosticsAnalyzer(
                    self.local_interp_model_name,
                )
            out["attention_diagnostics"] = self._attention_analyzer.analyze_prompt(
                full_prompt,
                layer_indices=layers,
                top_k=self.interp_top_k,
            )
        except Exception as e:
            out["attention_error"] = str(e)

        return out
