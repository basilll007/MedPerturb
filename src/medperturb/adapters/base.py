"""Provider-neutral model adapter contract.

Evaluation code only ever sees ModelResponse; every provider-specific detail
(SDK, constrained-output mechanism, token accounting, pricing) stays inside
the concrete adapter.
"""

import hashlib
import json
from abc import ABC, abstractmethod

from medperturb.schemas.response import Example, ModelResponse, OutputSpec


class ModelAdapter(ABC):
    provider: str
    model_id: str

    @abstractmethod
    def inference_config(self) -> dict:
        """Every parameter that can change the model's output. Part of the cache key."""

    @abstractmethod
    def generate(self, example: Example, prompt: str, output_spec: OutputSpec) -> ModelResponse:
        """One call. Must not raise for model-side outcomes (truncation etc.); may raise for transport errors."""

    def config_hash(self, prompt_version: str, cache_schema: str) -> str:
        payload = {
            "provider": self.provider, "model": self.model_id,
            "inference": self.inference_config(), "prompt_version": prompt_version, "cache_schema": cache_schema,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def json_schema_for(spec: OutputSpec) -> dict:
    """Provider-neutral JSON schema describing the expected answer object."""
    if spec.kind == "label":
        answer = {"type": "string", "enum": list(spec.allowed_labels)}
    elif spec.kind == "label_set":
        answer = {"type": "array", "items": {"type": "string", "enum": list(spec.allowed_labels)}}
    else:
        answer = {"type": "string"}
    return {"type": "object", "properties": {"answer": answer}, "required": ["answer"]}
