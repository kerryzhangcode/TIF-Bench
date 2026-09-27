"""Load forecast models from config by name."""

from __future__ import annotations

from typing import Any

from tifbench.models.gpt_astra import GptAstraModel
from tifbench.models.jev import DECISIONS_MODEL_NAMES, DecisionsForecastModel
from tifbench.models.openrouter_chat import OpenRouterChatModel

_OPENROUTER_CHAT = frozenset({
    "claude_opus_5_5",
    "gemma_3_12b",
    "qwen3_5_9b",
})


def get_model(name: str, cfg: dict[str, Any]):
    if name in DECISIONS_MODEL_NAMES:
        return DecisionsForecastModel(cfg, name=name)
    if name == "gpt_astra":
        return GptAstraModel(cfg)
    if name in _OPENROUTER_CHAT:
        return OpenRouterChatModel(cfg, name=name)
    raise KeyError(f"Unknown model: {name}")
