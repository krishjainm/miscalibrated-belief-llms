WMAC 2026 Prompt Pipeline
================================

Purpose
--------------------------------
This module provides a clean, WMAC 2026–aligned prompt pipeline for poker multi-agent experiments. It plugs into the existing `newRepOpenAI` game, logging, and analysis infrastructure by monkey‑patching the prompt generators used by LLM agents at runtime.

Key pieces
--------------------------------
- `prompt_schema.py`: Typed prompt I/O contracts to keep prompts structured and consistent
- `prompt_library.py`: All WMAC‑aligned prompts (roles, coordination, emergent, analysis) in one place
- `prompt_pipeline.py`: Assembles prompts from schema + game state into final strings
- `wmac_agents.py`: Light wrappers if needed (not required for initial integration)
- `run_wmac.py`: Runner that monkey‑patches the prompt functions and executes a simulation

Design goals
--------------------------------
- Research‑grade clarity and reproducibility
- Explicit guardrails for legal actions and amount validity
- Separation of concerns: prompt text vs. assembly vs. execution
- Zero intrusive edits to the rest of the codebase

Quick start
--------------------------------
From `newRepOpenAI/`:

```bash
python3 wmac2026/run_wmac.py \
  --num-hands 10 \
  --model gpt-3.5-turbo \
  --llm-players 0 1 2 3 \
  --collusion-llm-players 0 1 \
  --coordination-mode emergent_only
```

This will:
- Replace the prompt generators used by LLM agents with WMAC‑aligned versions
- Run a game using existing logging (EnhancedPromptLogger if available)
- Save outputs under `data/` as usual

Notes
--------------------------------
- You can iterate on prompts centrally in `prompt_library.py` without touching agent code
- The pipeline enforces action validity messaging (AVAILABLE ACTIONS + exact choice)
- Extend with additional roles or ablations by adding functions to `prompt_library.py` and wiring in `prompt_pipeline.py`


