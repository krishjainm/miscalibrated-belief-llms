"""
Logit lens: unembed hidden states to compare internal "next-token" predictions.

Requires ``transformers`` + ``torch`` and a local HuggingFace causal LM
(Mistral, Qwen, Llama, etc.). Not applicable to closed APIs.

Reference: nostalgebraist's logit lens observation; formalized in subsequent work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class LensStep:
    layer_index: int
    top_token_ids: list[int]
    top_token_strs: list[str]
    top_probs: list[float]
    entropy: float


class LogitLensAnalyzer:
    """
    Run a local causal LM and project a chosen layer's last-position hidden state
    through the language-model head to obtain token probabilities.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | dict = "auto",
        torch_dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device_map = device_map
        self._torch_dtype = torch_dtype

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self._torch_dtype, torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self._device_map,
            torch_dtype=dtype,
        )
        self._model.eval()

    def _lm_head_project(self, hidden_last_token: Any) -> Any:
        """Apply final norm (if present) + language-model head to last-token hidden state."""
        m = self._model
        h = hidden_last_token
        if hasattr(m, "model") and hasattr(m.model, "norm"):
            if h.dim() == 1:
                h = h.view(1, 1, -1)
            h = m.model.norm(h)
        # lm_head expects (batch, seq, hidden)
        if hasattr(m, "lm_head"):
            if h.dim() == 1:
                h = h.view(1, 1, -1)
            elif h.dim() == 2:
                h = h.unsqueeze(0)
            return m.lm_head(h).reshape(-1)
        if hasattr(m, "get_output_embeddings"):
            emb = m.get_output_embeddings()
            if h.dim() == 1:
                h = h.view(1, 1, -1)
            return emb(h).reshape(-1)
        raise RuntimeError("Cannot find lm_head or output embeddings on model")

    def analyze_prompt(
        self,
        prompt: str,
        *,
        layer_indices: Optional[list[int]] = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Forward pass with hidden states; for each layer in layer_indices,
        project last-token hidden state to logits and softmax top-k.

        Args:
            prompt: Full prompt text
            layer_indices: Layers to inspect (default: last layer only)
            top_k: Top tokens per layer

        Returns:
            Dict with keys ``layers`` (list of LensStep as dict) and ``model_name``.
        """
        self._lazy_load()
        import torch

        if layer_indices is None:
            layer_indices = [-1]

        enc = self._tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self._model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)

        hs = out.hidden_states  # tuple: embeddings + each layer
        steps: list[LensStep] = []

        for li in layer_indices:
            h_state = hs[li][0, -1, :]
            # Rebuild full sequence hidden for norm - use last position only
            logits = self._lm_head_project(h_state)
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, k=min(top_k, probs.shape[-1]))
            ids = top_i.cpu().numpy().tolist()
            pv = top_p.cpu().numpy().tolist()
            tok_strs = [self._tokenizer.decode([i]) for i in ids]
            ent = float(-(probs * (probs + 1e-12).log()).sum().item())

            steps.append(
                LensStep(
                    layer_index=li if li >= 0 else len(hs) + li,
                    top_token_ids=ids,
                    top_token_strs=tok_strs,
                    top_probs=pv,
                    entropy=ent,
                )
            )

        return {
            "model_name": self.model_name,
            "layers": [s.__dict__ for s in steps],
        }


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) for discrete distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))
