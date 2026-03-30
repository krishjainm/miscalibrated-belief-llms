"""
LLM agents and multi-provider inference for belief modeling experiments.

See run_llm_experiment.py for the main CLI.
"""

from llm.model_registry import ModelPreset, list_presets, resolve_preset
from llm.llm_agent import LLMAgent

__all__ = ["LLMAgent", "ModelPreset", "list_presets", "resolve_preset"]
