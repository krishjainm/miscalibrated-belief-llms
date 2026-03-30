# `llm_extra` JSON schema (decision logs)

This document is intended for **paper appendices** and **downstream analysis code**. It describes the optional field **`llm_extra`** on each **decision** record written to JSONL when using `LLMAgent` via `run_llm_experiment.py` (and the same structure is passed through `run_experiment.py` when the acting agent exposes `last_belief_meta` / `last_action_meta`).

Non-LLM agents (e.g. `RandomAgent`, `CallAgent`) omit `llm_extra` or set it to `null`.

## Top-level shape

```json
{
  "belief": { },
  "action": { }
}
```

- **`belief`**: Metadata from the **belief elicitation** API call (`LLMAgent.belief`), executed **after** `act` for the same decision point (two independent calls; order is fixed in `run_experiment.run_single_hand`).
- **`action`**: Metadata from the **action selection** API call (`LLMAgent.act`).

Either sub-object may be empty `{}` if the agent does not implement the corresponding path (not the case for `LLMAgent`).

---

## `belief` object (typical keys)

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | `boolean` | `true` if a valid 14-bucket distribution was parsed and normalized. |
| `parse_error` | `string` or `null` | Machine-readable parse failure tag (e.g. `no_json`, `missing_buckets:...`) or `null` on success. |
| `preset` | `string` | Model preset name from `llm/model_registry.py` (e.g. `gpt-4o-mini`). |
| `belief_mode` | `string` | `"direct"` or `"cot"` (see `analysis/cot_prompts.py`). |
| `raw_response_chars` | `integer` | Length of the raw model text (not the full raw string; avoids huge logs). |
| `has_logprobs` | `boolean` | Whether `logprobs` is non-null. |
| `logprobs` | `array` or `null` | Per-output-token logprob payload when the provider returns it (see **Providers and `logprobs_note`**). |
| `logprobs_note` | `string` or absent | Explains missing or partial logprobs (e.g. Anthropic has no API logprobs; Gemini install or model support). |
| `local_interp` | `object` or absent | Present only when local interpretability is enabled; see **Local interpretability** below. |
| `error` | `string` | Present if the belief API call raised (then `ok` is `false` and `belief` field on the decision may be `null`). |

---

## `action` object (typical keys)

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | `boolean` | `true` if the chosen action was parsed from JSON and matched a legal action. |
| `preset` | `string` | Same as belief. |
| `parse_fallback` | `boolean` | `true` if parsing failed and a deterministic fallback action was used. |
| `action_mode` | `string` | `"direct"` or `"cot"` (see `llm/prompts.py`). |
| `has_logprobs` | `boolean` | Whether `logprobs` is non-null. |
| `logprobs` | `array` or `null` | Same structure as belief. |
| `logprobs_note` | `string` or absent | Same as belief. |
| `local_interp` | `object` or absent | Same as belief, with `kind: "action"`. |
| `error` | `string` | Present if the action API call raised. |

---

## `logprobs` array (shared shape across providers)

When `run_llm_experiment.py` is run with `--top-logprobs K` (`K > 0`), the client requests logprobs where the API supports it. We serialize **one canonical per-token shape** so analysis code can treat OpenAI-compatible chat, DashScope Qwen, and Gemini (via `google-genai`) the same way:

```json
[
  {
    "token": "<string|null>",
    "logprob": <float|null>,
    "bytes": <array|null>,
    "top_logprobs": { "<token_string>": <logprob_float>, ... } | null
  },
  ...
]
```

**Semantics**

- The array is ordered **left-to-right** over **generated completion tokens** (as returned by the API’s `choice.logprobs.content`).
- **`token`**: The sampled token at this position (string form as returned by the API).
- **`logprob`**: Natural log probability of the sampled token (API field).
- **`bytes`**: Optional byte representation (when the API provides it); may be `null`.
- **`top_logprobs`**: Map from **alternative token strings** to **logprob** for up to `K` candidates at this position (excluding or including the sampled token depending on provider; treat as provider-specific).

### Providers and `logprobs_note`

| Provider / preset family | `logprobs` when `--top-logprobs` | Notes |
|--------------------------|----------------------------------|--------|
| `openai`, `openai_compatible` (Mistral, Together, **DashScope Qwen**) | Filled when the endpoint returns Chat Completions `logprobs` | DashScope: set `DASHSCOPE_API_KEY`; optional `DASHSCOPE_BASE_URL` overrides the preset `base_url` (see Alibaba Model Studio). |
| `anthropic` | Always `null` | `logprobs_note` explains that the Messages API does not expose output token logprobs. |
| `google` | Filled when `google-genai` succeeds and the model returns `logprobs_result` | Uses `GenerateContentConfig(response_logprobs=True, logprobs=K)` (clamped to 1–20). If `google-genai` is missing, logprobs fail, or the response has no `logprobs_result`, we fall back to legacy `google.generativeai` and set `logprobs_note` accordingly. Without `--top-logprobs`, Google uses the legacy client only (no logprobs). |

**Interpretation note for papers:** These are **conditional next-token distributions over the model’s tokenizer**, not probabilities over poker bucket labels. They are useful as a **decision-time uncertainty / sampling pressure** signal, not as a substitute for elicited bucket beliefs.

---

## Local interpretability (`local_interp`)

Enabled when `run_llm_experiment.py` is run with:

- `--local-interp-model <HF_MODEL_ID>` (e.g. `mistralai/Mistral-7B-Instruct-v0.3`)
- `--interp-max-calls N` with `N > 0`

Requires **`torch`**, **`transformers`**, and usually **`accelerate`** (see `requirements-local.txt` and the setup section below).

The agent concatenates the same **system** and **user** strings used for that call into:

`full_prompt = "<system>\\n\\n<user>"`

and runs local analyzers on `full_prompt`. Each successful run increments a shared counter until `interp_max_calls` is reached; further decisions omit new `local_interp` or return `ran: false`.

### `local_interp` object (success / partial)

| Key | Type | Meaning |
|-----|------|---------|
| `ran` | `boolean` | `true` if this call counted against `interp_max_calls`. |
| `kind` | `string` | `"belief"` or `"action"`. |
| `layers` | `array` of `integer` | Layer indices passed to analyzers (e.g. `[-1, -2]`). |
| `logit_lens` | `object` | Output of `LogitLensAnalyzer.analyze_prompt` (see below). |
| `logit_lens_error` | `string` | Present if logit lens failed (e.g. missing deps, OOM). |
| `attention_diagnostics` | `object` | Output of `AttentionDiagnosticsAnalyzer.analyze_prompt` (see below). |
| `attention_error` | `string` | Present if attention diagnostics failed. |

### `logit_lens` object

```json
{
  "model_name": "<HF model id>",
  "layers": [
    {
      "layer_index": <int>,
      "top_token_ids": [<int>, ...],
      "top_token_strs": [<string>, ...],
      "top_probs": [<float>, ...],
      "entropy": <float>
    }
  ]
}
```

- **`layer_index`**: Index into `outputs.hidden_states` from Hugging Face (negative indices are resolved in code to `len(hidden_states) + index` for the stored value in `LensStep`).
- **`top_probs` / `top_token_strs`**: Top-`interp_top_k` tokens under a softmax over **vocabulary** after projecting the **last prompt position** hidden state through the model LM head (logit lens–style readout).
- **`entropy`**: Shannon entropy of the **full** vocabulary distribution at that projection (not truncated to top-k).

### `attention_diagnostics` object

```json
{
  "model_name": "<HF model id>",
  "layers": [
    {
      "layer_index": <int>,
      "entropy": <float>,
      "top_key_positions": [<int>, ...],
      "top_key_tokens": [<string>, ...],
      "top_attention": [<float>, ...]
    }
  ]
}
```

- For each requested layer, we take attention at the **last query index**, **average over heads**, normalize over keys to a distribution over prompt positions, then report **entropy** and top-`interp_top_k` key positions (decoded token pieces for those positions).

### `local_interp` when budget exhausted

```json
{ "ran": false, "reason": "interp_max_calls_reached" }
```

---

## Related top-level decision fields (not inside `llm_extra`)

For cross-reference in appendices:

| Field | Meaning |
|-------|---------|
| `agent_belief` | Parsed bucket distribution (`dict` of 14 keys) or `null` if parse failed / no belief. |
| `agent_action` | `FOLD`, `CHECK_OR_CALL`, or `BET_OR_RAISE`. |
| `prompt_version` | Stored on the **`run_config`** header line (`type: "run_config"`), not on each decision; encodes belief/action prompt family (see `run_llm_experiment.py`). |

---

## Installing local `torch` + `transformers` (manual / environment-specific)

GPU memory and CUDA wheels vary by machine; these steps are intentionally **manual**.

1. **Use the same Python as your venv** (3.11+).

   ```bash
   cd poker2026-main
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Unix:
   # source venv/bin/activate
   ```

2. **Install base + API LLM deps** (if not already):

   ```bash
   pip install -r requirements.txt -r requirements-llm.txt
   ```

3. **Install local interpretability stack** (pick ONE torch channel that matches your hardware):

   **CPU-only (slow, for smoke tests):**

   ```bash
   pip install -r requirements-local.txt --index-url https://download.pytorch.org/whl/cpu
   ```

   **CUDA (example: CUDA 12.1 wheels — adjust to your driver/toolkit):**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements-local.txt
   ```

   If you already have PyTorch installed correctly, you can skip the torch line and only run `pip install -r requirements-local.txt`.

4. **Authenticate Hugging Face** if the checkpoint is gated:

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

5. **Run an experiment with local interp** (example):

   **Unix / bash** (line continuation with `\`):

   ```bash
   python run_llm_experiment.py --preset mistral-small --belief-mode direct --action-mode direct \
     --local-interp-model mistralai/Mistral-7B-Instruct-v0.3 \
     --interp-max-calls 20 --interp-layers -1,-2 --interp-top-k 10 \
     --hands 5 --no-oracle -v --out logs/llm_local_interp.jsonl
   ```

   **Windows PowerShell:** do **not** use `^` (that is **cmd.exe** only). Use **one line**, or end each line with a **backtick** `` ` `` (no space after it). For `--interp-layers` when indices are negative, use **equals form** (`--interp-layers=-1,-2`). Otherwise argparse may treat `-1` as a new flag and error with `expected one argument`.

   ```powershell
   python run_llm_experiment.py --preset mistral-small --belief-mode direct --action-mode direct --local-interp-model mistralai/Mistral-7B-Instruct-v0.3 --interp-max-calls 20 --interp-layers=-1,-2 --interp-top-k 10 --hands 5 --no-oracle -v --out logs/llm_local_interp.jsonl
   ```

   Use a **small** `--interp-max-calls` first: each qualifying belief/action call can load the model once and run two forward passes (hidden states + attentions).

**Windows note:** If `bfloat16` fails on CPU, you may need to patch `LogitLensAnalyzer` / `AttentionDiagnosticsAnalyzer` to use `torch_dtype="float32"` (not exposed on CLI today); open an issue or set in code for your runs.
