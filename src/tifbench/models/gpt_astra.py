"""GPT-6 Astra formal forecast baseline via OpenAI Responses API (high effort)."""

from __future__ import annotations

from typing import Any

from tifbench.models.openai_responses_forecast import OpenAIResponsesForecastModel

PROMPT_VERSION = "forecast_gpt_astra"


class GptAstraModel(OpenAIResponsesForecastModel):
    """Formal comparison baseline: gpt-6-astra with reasoning.effort=high."""

    name = "gpt_astra"
    forecast_transport = "openai_batch"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg, config_key="gpt_astra", prompt_version=PROMPT_VERSION)
