"""
Optional mechanistic-interpretability directions for this research path.

These are interfaces and stubs you can extend for the updated paper:

- **Logit lens** (implemented in ``logit_lens.py``): compare internal predictions
  at intermediate layers when running local HF checkpoints.
- **Output-token logprobs**: OpenAI-compatible ``top_logprobs`` on the belief or
  action completion (see ``ChatClient.complete(..., top_logprobs=k)``).
- **Attention rollout** (stub): aggregate attention to input tokens for a decision.
- **Linear probes** (stub): train small probes on hidden states to predict bucket
  or action from representations (requires saved activations).
- **Contrastive activation patching** (stub): swap activations between prompts.

Closed models (Claude, Gemini, GPT) are limited to API-exposed information unless
you use a partner program that provides logprobs or batch inference with disclosure.
"""

from __future__ import annotations

from typing import Any, Protocol


class ActivationSaver(Protocol):
    """Hook interface for saving layer activations during a forward pass."""

    def save(self, layer: int, tensor: Any) -> None: ...


def attention_rollout_stub(_attention_weights: Any) -> Any:
    """
    Placeholder for attention rollout to input tokens.

    Implement with model-specific attention outputs if you add a forward hook
    that stores per-layer attention matrices.
    """
    raise NotImplementedError(
        "attention_rollout_stub: provide attention weights from a hooked model."
    )


def linear_probe_interface(
    hidden_states: Any,
    labels: Any,
) -> Any:
    """
    Placeholder for training a sklearn/torch linear probe on saved hiddens.

    Typical use: predict opponent bucket or action from last-token hidden state.
    """
    raise NotImplementedError(
        "linear_probe_interface: implement with your saved tensors and labels."
    )
