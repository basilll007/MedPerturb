"""Qwen local adapter for Unsloth 4-bit models.

Implements ModelAdapter contract so evaluation runner, caching, parsing,
metrics, and QC code treat Qwen identically to Gemini and Claude.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import torch

from medperturb.adapters.base import ModelAdapter
from medperturb.schemas.response import Example, ModelResponse, OutputSpec

DEFAULT_SYSTEM_PROMPT = (
    'Respond strictly with a JSON object in the form {"answer": "<label>"} or '
    '{"answer": ["<label>", ...]} or {"answer": "<option text>"}. '
    "Do not output any markdown formatting, explanation, or commentary."
)


class QwenAdapter(ModelAdapter):
    provider = "unsloth"

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_id: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        adapter_path: str | Path | None = None,
        max_seq_length: int = 2048,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            from unsloth import FastLanguageModel

            load_target = self.adapter_path if self.adapter_path is not None else self.model_id
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_target,
                max_seq_length=self.max_seq_length,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)

    def inference_config(self) -> dict[str, Any]:
        cfg = {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": True,
            "quantization": "bnb-4bit",
            "enable_thinking": False,
            "system_prompt": self.system_prompt,
        }
        if self.adapter_path is not None:
            cfg["adapter_path"] = self.adapter_path
        return cfg

    def unload(self) -> None:
        """Explicitly free model and tokenizer from VRAM."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def generate(self, example: Example, prompt: str, output_spec: OutputSpec) -> ModelResponse:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to("cuda")
        in_tok = int(inputs.input_ids.shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0

        gen_tokens = outputs[0][in_tok:]
        out_tok = int(len(gen_tokens))
        text_resp = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        finish_reason = "STOP"
        if out_tok >= self.max_new_tokens:
            finish_reason = "MAX_TOKENS"

        vram_allocated_gb = float(torch.cuda.memory_allocated() / (1024**3))

        # Enforce canonical JSON contract: ensure {"answer": ...} format
        raw_json_str = text_resp
        try:
            obj = json.loads(text_resp)
            if not isinstance(obj, dict) or "answer" not in obj:
                raw_json_str = json.dumps({"answer": text_resp})
        except Exception:
            # Handle markdown codeblock if present
            if text_resp.startswith("```"):
                lines = [l for l in text_resp.split("\n") if not l.startswith("```")]
                inner = "\n".join(lines).strip()
                try:
                    obj = json.loads(inner)
                    if isinstance(obj, dict) and "answer" in obj:
                        raw_json_str = inner
                    else:
                        raw_json_str = json.dumps({"answer": inner})
                except Exception:
                    raw_json_str = json.dumps({"answer": inner})
            else:
                raw_json_str = json.dumps({"answer": text_resp})

        return ModelResponse(
            provider=self.provider,
            model=self.model_id,
            model_version="bnb-4bit",
            question_id=example.question_id,
            condition=example.condition,
            raw_response=raw_json_str,
            finish_reason=finish_reason,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thinking_tokens=0,
            latency_s=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cost_usd=0.0,
            raw_payload={
                "vram_allocated_gb": round(vram_allocated_gb, 4),
                "unnormalized_text": text_resp,
            },
        )
