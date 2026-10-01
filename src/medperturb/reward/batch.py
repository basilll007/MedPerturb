"""Batch reward wrappers compatible with TRL GRPOTrainer and Unsloth."""

from __future__ import annotations

import json
from typing import Any

from medperturb.reward.evaluators import (
    compute_r_correct,
    compute_r_format,
    compute_r_symbolic,
    compute_reward,
)
from medperturb.schemas.response import Example


def extract_completion_text(completion: Any) -> str:
    """Extracts raw generated text from string or chat message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1]["content"])
        if completion and isinstance(completion[0], dict) and "content" in completion[0]:
            return str(completion[0]["content"])
    return str(completion)


def reconstruct_example(raw_ex: Any) -> Example:
    """Reconstructs Example object from dict, JSON string, or Example instance."""
    if isinstance(raw_ex, Example):
        return raw_ex
    if isinstance(raw_ex, str):
        d = json.loads(raw_ex)
    elif isinstance(raw_ex, dict):
        d = raw_ex
    else:
        raise TypeError(f"Cannot reconstruct Example from {type(raw_ex)}")

    return Example(
        question_id=d["question_id"],
        source_dataset=d["source_dataset"],
        source_id=d["source_id"],
        condition=d["condition"],
        question=d["question"],
        option_labels=d.get("option_labels", []),
        option_texts=d.get("option_texts", []),
        options_displayed=d.get("options_displayed", True),
        gold_label=d.get("gold_label"),
        gold_text=d.get("gold_text"),
        gold_index=d.get("gold_index"),
        gold_indices=d.get("gold_indices"),
    )


def reward_correct_func(prompts: list[Any], completions: list[Any], **kwargs) -> list[float]:
    """TRL-compatible reward function for exact task correctness (R_correct)."""
    examples_raw = kwargs.get("example_json") or kwargs.get("example")
    if not examples_raw:
        raise ValueError("Missing 'example_json' or 'example' in reward_kwargs")

    rewards = []
    for comp, ex_raw in zip(completions, examples_raw):
        ex = reconstruct_example(ex_raw)
        text = extract_completion_text(comp)
        r_corr, _, _ = compute_r_correct(ex, text)
        rewards.append(float(r_corr))
    return rewards


def create_reward_symbolic_func(lambda_symbolic: float = 1.0):
    """Creates a TRL-compatible reward function for symbolic transformation semantics (lambda * R_symbolic)."""

    def reward_symbolic_func(prompts: list[Any], completions: list[Any], **kwargs) -> list[float]:
        examples_raw = kwargs.get("example_json") or kwargs.get("example")
        if not examples_raw:
            raise ValueError("Missing 'example_json' or 'example' in reward_kwargs")

        rewards = []
        for comp, ex_raw in zip(completions, examples_raw):
            ex = reconstruct_example(ex_raw)
            text = extract_completion_text(comp)
            r_symb, _ = compute_r_symbolic(ex, text)
            rewards.append(float(lambda_symbolic * r_symb))
        return rewards

    reward_symbolic_func.__name__ = f"reward_symbolic_lambda_{lambda_symbolic}"
    return reward_symbolic_func


def reward_format_func(prompts: list[Any], completions: list[Any], **kwargs) -> list[float]:
    """TRL-compatible reward function for JSON output contract compliance."""
    rewards = []
    for comp in completions:
        text = extract_completion_text(comp)
        rewards.append(float(compute_r_format(text)))
    return rewards


def create_reward_ns_func(lambda_symbolic: float = 1.0, beta_format: float = 0.1):
    """Creates a single combined neuro-symbolic reward function: R_correct + lambda * R_symbolic + beta * R_format."""

    def reward_ns_func(prompts: list[Any], completions: list[Any], **kwargs) -> list[float]:
        examples_raw = kwargs.get("example_json") or kwargs.get("example")
        if not examples_raw:
            raise ValueError("Missing 'example_json' or 'example' in reward_kwargs")

        rewards = []
        for comp, ex_raw in zip(completions, examples_raw):
            ex = reconstruct_example(ex_raw)
            text = extract_completion_text(comp)
            breakdown = compute_reward(ex, text, lambda_symbolic=lambda_symbolic, beta_format=beta_format)
            rewards.append(float(breakdown.r_total))
        return rewards

    reward_ns_func.__name__ = f"reward_ns_l{lambda_symbolic}_b{beta_format}"
    return reward_ns_func
