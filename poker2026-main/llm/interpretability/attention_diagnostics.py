"""
Attention diagnostics for local causal LMs.

We compute layerwise attention entropy at the *last query position*:
  entropy( mean_over_heads( attention[last_query_pos, :] ) )

This is a lightweight alternative interpretability signal to logit lens:
it probes how "peaky" attention is over the prompt at different depths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class AttentionStep:
    layer_index: int
    entropy: float
    top_key_positions: list[int]
    top_key_tokens: list[str]
    top_attention: list[float]


class AttentionDiagnosticsAnalyzer:
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

    def analyze_prompt(
        self,
        prompt: str,
        *,
        layer_indices: Optional[list[int]] = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        self._lazy_load()
        import torch

        if layer_indices is None:
            layer_indices = [-1]

        enc = self._tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self._model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self._model(**enc, output_attentions=True)

        attentions = out.attentions  # tuple of (batch, heads, q_len, k_len)

        steps: list[AttentionStep] = []
        # attentions length usually equals num_layers
        for li in layer_indices:
            att = attentions[li]  # (B, H, Q, K)
            # last query position for each layer
            # (B, H, K)
            att_last = att[:, :, -1, :]
            # mean over heads and batch (batch is 1)
            att_vec = att_last.mean(dim=1).squeeze(0)  # (K,)
            att_vec = att_vec / (att_vec.sum() + 1e-12)

            probs = att_vec.detach().cpu().numpy()
            entropy = float(-(probs * (np.log(probs + 1e-12))).sum())

            k = min(top_k, probs.shape[0])
            top_idx = np.argsort(probs)[::-1][:k].tolist()
            tok_ids = [int(enc["input_ids"][0, i].detach().cpu().item()) for i in top_idx]
            tok_strs = [self._tokenizer.decode([tid]) for tid in tok_ids]
            top_att = [float(probs[i]) for i in top_idx]

            steps.append(
                AttentionStep(
                    layer_index=li,
                    entropy=entropy,
                    top_key_positions=top_idx,
                    top_key_tokens=tok_strs,
                    top_attention=top_att,
                )
            )

        return {
            "model_name": self.model_name,
            "layers": [s.__dict__ for s in steps],
        }

