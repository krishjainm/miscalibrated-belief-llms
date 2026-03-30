"""
Mechanistic interpretability tools for local (HF) models.

- logit_lens: project hidden states to vocabulary logits
- mechanistic: registry of optional techniques (attention rollout stubs, etc.)

API-only models (GPT, Claude, Gemini) do not expose hidden states; use
``top_logprobs`` on the completion API where supported (OpenAI-compatible).
"""

from llm.interpretability.logit_lens import LogitLensAnalyzer
from llm.interpretability.attention_diagnostics import AttentionDiagnosticsAnalyzer

__all__ = ["LogitLensAnalyzer", "AttentionDiagnosticsAnalyzer"]
