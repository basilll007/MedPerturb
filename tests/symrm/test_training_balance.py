"""Unit test confirming training data class balance for Phase 2.

Verifies:
- Loss weighting: edited items weight 2.0, base and near-miss items weight 1.0.
- Effective class balance (default-chosen vs alternative-chosen) == 50/50.
"""

import json
from pathlib import Path
import pytest
from symrm.config import SymRMConfig


def test_training_loss_weighting_and_class_balance():
    # 1. Config validation
    cfg = SymRMConfig.from_yaml("configs/symrm/default.yaml")
    assert cfg.training.balanced_loss_weighting is True
    assert cfg.training.sample_weight_base == 1.0
    assert cfg.training.sample_weight_near_miss == 1.0
    assert cfg.training.sample_weight_edited == 2.0

    # 2. Dataset validation
    data_path = Path("results/symrm/data/train.jsonl")
    assert data_path.exists(), f"Train dataset not found at {data_path}"

    with open(data_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    assert len(items) > 0

    weight_default = 0.0
    weight_alternative = 0.0

    for it in items:
        itype = it["item_type"]
        if itype == "base":
            w = cfg.training.sample_weight_base
        elif itype == "near_miss":
            w = cfg.training.sample_weight_near_miss
        elif itype == "edited":
            w = cfg.training.sample_weight_edited
        else:
            w = 1.0

        if it["chosen_is_default"]:
            weight_default += w
        else:
            weight_alternative += w

    total_weight = weight_default + weight_alternative
    pct_default = weight_default / total_weight
    pct_alternative = weight_alternative / total_weight

    assert abs(pct_default - 0.50) < 1e-6, f"Effective default weight {pct_default} != 0.50"
    assert abs(pct_alternative - 0.50) < 1e-6, f"Effective alternative weight {pct_alternative} != 0.50"
