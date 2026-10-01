"""Configuration schema and validation for SymRM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Optional
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-1.5B"
    revision: Optional[str] = None
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"
    quant_type: str = "nf4"
    max_seq_length: int = 512
    padding_strategy: Literal["dynamic", "max_length"] = "dynamic"
    device_map: str = "auto"


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"
    task_type: str = "SEQ_CLS"


class DataConfig(BaseModel):
    rules_file: str = "configs/symrm/rules.yaml"
    target_pairs: int = 300
    train_families: List[str] = Field(default_factory=lambda: ["renal", "pregnancy"])
    heldout_family: str = "allergy"
    near_ood_rule_ids: List[str] = Field(default_factory=lambda: ["renal_02", "preg_02"])
    near_miss_ratio: float = 0.5
    seed: int = 42
    output_dir: str = "results/symrm/data"


class ProbeConfig(BaseModel):
    target_rms: List[str] = Field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-0.5B",
        ]
    )
    layers: Optional[List[int]] = None  # None = probe multiple representative layers
    cv_folds: int = 5
    seed: int = 42
    output_dir: str = "results/symrm/probe"


class TrainingConfig(BaseModel):
    condition: Literal["c1_outcome", "c2_counterfactual", "c3_symbolic", "c4_both"] = "c1_outcome"
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    padding_strategy: Literal["dynamic", "max_length"] = "dynamic"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    seeds: List[int] = Field(default_factory=lambda: [42, 43, 44])
    early_stopping_patience: int = 2
    pair_consistency_weight: float = 1.0
    symbolic_aux_weight: float = 1.0
    balanced_loss_weighting: bool = True
    sample_weight_base: float = 1.0
    sample_weight_near_miss: float = 1.0
    sample_weight_edited: float = 2.0
    output_dir: str = "results/symrm/checkpoints"


class BestOfNConfig(BaseModel):
    policy_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    n_candidates: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 128
    eval_n_values: List[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    seed: int = 42
    output_dir: str = "results/symrm/best_of_n"


class LoggingConfig(BaseModel):
    results_dir: str = "results/symrm"
    compute_log_path: str = "results/symrm/compute_log.csv"
    preregistration_file: str = "results/symrm/preregistration.md"


class SymRMConfig(BaseModel):
    phase: Literal["gate0", "phase1_data", "phase1_probe", "phase2_train", "phase3_bon"] = "gate0"
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    best_of_n: BestOfNConfig = Field(default_factory=BestOfNConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SymRMConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, sort_keys=False, indent=2)
