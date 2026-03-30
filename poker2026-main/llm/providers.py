"""
Multi-provider chat completion: OpenAI, Anthropic, Google, OpenAI-compatible (Mistral, Together, DashScope).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from llm.model_registry import ModelPreset


@dataclass
class ChatResult:
    text: str
    raw: Any | None = None
    logprobs: list[dict] | None = None
    logprobs_note: str | None = None


def _serialize_openai_style_logprobs(choice: Any) -> list[dict] | None:
    if not getattr(choice, "logprobs", None) or not getattr(choice.logprobs, "content", None):
        return None
    try:
        per_pos = []
        for lp in choice.logprobs.content:
            top_map = None
            if getattr(lp, "top_logprobs", None):
                try:
                    top_map = {t.token: t.logprob for t in lp.top_logprobs}
                except Exception:
                    top_map = None
            per_pos.append(
                {
                    "token": getattr(lp, "token", None),
                    "logprob": getattr(lp, "logprob", None),
                    "bytes": getattr(lp, "bytes", None),
                    "top_logprobs": top_map,
                }
            )
        return per_pos
    except Exception:
        return None


def _serialize_gemini_logprobs_result(lr: Any) -> list[dict] | None:
    """Map Gemini LogprobsResult to the same per-position shape as OpenAI."""
    if lr is None:
        return None
    chosen = getattr(lr, "chosen_candidates", None) or []
    tops = getattr(lr, "top_candidates", None) or []
    if not chosen and not tops:
        return None
    n = max(len(chosen), len(tops))
    per_pos: list[dict] = []
    for i in range(n):
        ch = chosen[i] if i < len(chosen) else None
        top_list = tops[i].candidates if i < len(tops) and tops[i] and tops[i].candidates else []
        top_map: dict[str, float] = {}
        for c in top_list:
            tok = getattr(c, "token", None)
            lpv = getattr(c, "log_probability", None)
            if tok is not None and lpv is not None:
                top_map[str(tok)] = float(lpv)
        per_pos.append(
            {
                "token": getattr(ch, "token", None) if ch else None,
                "logprob": float(getattr(ch, "log_probability")) if ch and getattr(ch, "log_probability", None) is not None else None,
                "bytes": None,
                "top_logprobs": top_map if top_map else None,
            }
        )
    return per_pos if per_pos else None


class ChatClient:
    """Unified minimal chat interface for experiments."""

    def __init__(self, preset: ModelPreset):
        self.preset = preset
        self._api_key = os.environ.get(preset.api_key_env, "")

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_logprobs: int | None = None,
    ) -> ChatResult:
        if not self._api_key:
            raise RuntimeError(
                f"Missing API key: set environment variable {self.preset.api_key_env}"
            )

        if self.preset.provider == "openai":
            return self._openai(system, user, temperature, max_tokens, top_logprobs)
        if self.preset.provider == "openai_compatible":
            return self._openai_compatible(system, user, temperature, max_tokens, top_logprobs)
        if self.preset.provider == "anthropic":
            return self._anthropic(system, user, temperature, max_tokens, top_logprobs)
        if self.preset.provider == "google":
            return self._google(system, user, temperature, max_tokens, top_logprobs)
        raise ValueError(f"Unsupported provider: {self.preset.provider}")

    def _compatible_base_url(self) -> str:
        base = self.preset.base_url or "https://api.openai.com/v1"
        if self.preset.api_key_env == "DASHSCOPE_API_KEY":
            base = os.environ.get("DASHSCOPE_BASE_URL") or base
        return base

    def _openai(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_logprobs: int | None,
    ) -> ChatResult:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        kwargs: dict = {
            "model": self.preset.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_logprobs is not None and top_logprobs > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        logprobs = _serialize_openai_style_logprobs(choice)
        return ChatResult(text=text, raw=resp, logprobs=logprobs)

    def _openai_compatible(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_logprobs: int | None,
    ) -> ChatResult:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._compatible_base_url())
        kwargs: dict = {
            "model": self.preset.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_logprobs is not None and top_logprobs > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        logprobs = _serialize_openai_style_logprobs(choice)
        return ChatResult(text=text, raw=resp, logprobs=logprobs)

    def _anthropic(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_logprobs: int | None,
    ) -> ChatResult:
        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key)
        msg = client.messages.create(
            model=self.preset.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        note = None
        if top_logprobs is not None and top_logprobs > 0:
            note = (
                "anthropic: Claude Messages API does not expose output token logprobs; "
                "use an OpenAI-compatible preset with --top-logprobs, or compare without logprobs."
            )
        return ChatResult(text="".join(parts), raw=msg, logprobs=None, logprobs_note=note)

    def _google(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_logprobs: int | None,
    ) -> ChatResult:
        if top_logprobs is not None and top_logprobs > 0:
            try:
                return self._google_with_logprobs(
                    system, user, temperature, max_tokens, top_logprobs
                )
            except ImportError:
                return self._google_legacy(
                    system, user, temperature, max_tokens,
                    logprobs_note=(
                        "google: install package google-genai (pip install google-genai) "
                        "to populate logprobs for Gemini when using --top-logprobs"
                    ),
                )
            except Exception as e:
                return self._google_legacy(
                    system, user, temperature, max_tokens,
                    logprobs_note=f"google_genai_logprobs_failed: {e!s}",
                )
        return self._google_legacy(system, user, temperature, max_tokens)

    def _google_with_logprobs(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_logprobs: int,
    ) -> ChatResult:
        from google import genai
        from google.genai import types

        k = max(1, min(20, int(top_logprobs)))
        client = genai.Client(api_key=self._api_key)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_logprobs=True,
            logprobs=k,
        )
        resp = client.models.generate_content(
            model=self.preset.model_id,
            contents=user,
            config=config,
        )
        text = (resp.text or "").strip() if getattr(resp, "text", None) else ""
        if not text and resp.candidates:
            for part in resp.candidates[0].content.parts:
                text += getattr(part, "text", "") or ""
        logprobs = None
        note = None
        if resp.candidates:
            lr = getattr(resp.candidates[0], "logprobs_result", None)
            logprobs = _serialize_gemini_logprobs_result(lr)
            if logprobs is None:
                note = "google: response had no logprobs_result (model or region may not support response_logprobs)"
        else:
            note = "google: empty candidates in response"
        return ChatResult(text=text, raw=resp, logprobs=logprobs, logprobs_note=note)

    def _google_legacy(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        logprobs_note: str | None = None,
    ) -> ChatResult:
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(
            self.preset.model_id,
            system_instruction=system,
        )
        resp = model.generate_content(
            user,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = ""
        if resp.candidates:
            for part in resp.candidates[0].content.parts:
                text += getattr(part, "text", "") or ""
        return ChatResult(text=text, raw=resp, logprobs=None, logprobs_note=logprobs_note)


def make_client(preset: ModelPreset) -> ChatClient:
    return ChatClient(preset)
