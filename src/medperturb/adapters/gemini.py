"""Gemini adapter. All google-genai specifics live here."""

import json
import time
from datetime import datetime, timezone

from medperturb.adapters.base import ModelAdapter, json_schema_for
from medperturb.schemas.response import Example, ModelResponse, OutputSpec

# USD per 1M tokens, sourced 2026-09-29 from ai.google.dev/gemini-api/docs/pricing
# (promotional rate through 2026-12-31). Thinking tokens are billed as output.
PRICING = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75},
}


def _to_dict(obj):
    try:
        return obj.model_dump(mode="json")
    except Exception:
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return {"repr": str(obj)}


class GeminiAdapter(ModelAdapter):
    provider = "gemini"

    def __init__(self, model_id: str, api_key: str, thinking_level: str = "low", max_output_tokens: int = 1536):
        from google import genai
        self.model_id = model_id
        self._client = genai.Client(api_key=api_key)
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens

    def inference_config(self) -> dict:
        return {
            "thinking_level": self.thinking_level,
            "max_output_tokens": self.max_output_tokens,
            "max_output_tokens_semantics": "shared ceiling for thinking + visible tokens (per Gemini docs)",
            "structured_output": "response_mime_type=application/json + response_json_schema",
        }

    def cost(self, input_tokens, output_tokens, thinking_tokens):
        p = PRICING.get(self.model_id)
        if p is None or input_tokens is None:
            return None
        billed_out = (output_tokens or 0) + (thinking_tokens or 0)
        return input_tokens / 1e6 * p["input"] + billed_out / 1e6 * p["output"]

    def generate(self, example: Example, prompt: str, output_spec: OutputSpec) -> ModelResponse:
        from google.genai import types

        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            # Gemini 3.x: temperature/top_p/top_k are deprecated and ignored, and the
            # string thinking_level replaces the numeric thinking_budget - sending both
            # in one request is a 400. "minimal" is not supported on 3.7/3.8-flash.
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
                response_mime_type="application/json",
                response_json_schema=json_schema_for(output_spec),
            ),
        )
        latency = time.perf_counter() - t0

        try:
            text = resp.text
        except Exception:
            text = None
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) if usage else None
        out_tok = getattr(usage, "candidates_token_count", None) if usage else None
        think_tok = getattr(usage, "thoughts_token_count", None) if usage else None
        try:
            finish = resp.candidates[0].finish_reason
            finish = getattr(finish, "name", None) or str(finish)
        except (AttributeError, IndexError, TypeError):
            finish = None

        return ModelResponse(
            provider=self.provider, model=self.model_id,
            model_version=getattr(resp, "model_version", None),
            question_id=example.question_id, condition=example.condition,
            raw_response=text, finish_reason=finish,
            input_tokens=in_tok, output_tokens=out_tok, thinking_tokens=think_tok,
            latency_s=latency, timestamp=datetime.now(timezone.utc).isoformat(),
            cost_usd=self.cost(in_tok, out_tok, think_tok), raw_payload=_to_dict(resp),
        )
