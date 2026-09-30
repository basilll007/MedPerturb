"""Config-hashed response cache. A cached result is reused only if it was
produced under the identical adapter/inference/prompt configuration AND the
identical rendered prompt; anything else is an incompatibility, never a
silent reuse."""

import json
from dataclasses import asdict
from pathlib import Path

from medperturb.schemas.response import ModelResponse

CACHE_SCHEMA = "phase2a_cache_v1"


class CacheIncompatible(RuntimeError):
    pass


class ResponseCache:
    def __init__(self, root: Path, config_hash: str, secret_values: list[str]):
        self.root = root / config_hash[:16]
        self.config_hash = config_hash
        self._secrets = [s for s in secret_values if s]

    def _path(self, question_id: str, condition: str) -> Path:
        safe = question_id.replace("::", "__").replace("/", "_")
        return self.root / condition / f"{safe}.json"

    def get(self, question_id: str, condition: str, prompt_sha: str) -> ModelResponse | None:
        p = self._path(question_id, condition)
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("config_hash") != self.config_hash or rec.get("cache_schema") != CACHE_SCHEMA:
            raise CacheIncompatible(f"{p}: produced under a different configuration")
        if rec.get("prompt_sha256") != prompt_sha:
            raise CacheIncompatible(f"{p}: rendered prompt changed under the same prompt_version - bump the version")
        return ModelResponse(**rec["response"])

    def put(self, resp: ModelResponse, prompt: str, prompt_sha: str, prompt_version: str):
        rec = {
            "cache_schema": CACHE_SCHEMA, "config_hash": self.config_hash,
            "prompt_version": prompt_version, "prompt_sha256": prompt_sha, "prompt": prompt,
            "response": asdict(resp),
        }
        text = json.dumps(rec, indent=2, default=str)
        for s in self._secrets:
            if s in text:
                raise RuntimeError("secret value found in response record - refusing to write")
        p = self._path(resp.question_id, resp.condition)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)  # atomic: an interrupted write never leaves a half-valid cache entry
