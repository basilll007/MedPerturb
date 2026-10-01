"""Unit tests for MedPerturb Phase 3 Symbolic and Correctness Rewards."""

import pytest
from medperturb.reward.evaluators import (
    compute_r_correct,
    compute_r_format,
    compute_r_symbolic,
    compute_reward,
)
from medperturb.schemas.response import Example


@pytest.fixture
def sample_mcq_example():
    return Example(
        question_id="test::001",
        source_dataset="test",
        source_id="001",
        condition="mcq",
        question="What is the primary treatment for streptococcal pharyngitis?",
        option_labels=["A", "B", "C", "D"],
        option_texts=["Amoxicillin", "Vancomycin", "Ciprofloxacin", "Acyclovir"],
        options_displayed=True,
        gold_label="A",
        gold_text="Amoxicillin",
        gold_index=0,
    )


@pytest.fixture
def sample_roman_example():
    return Example(
        question_id="test::001",
        source_dataset="test",
        source_id="001",
        condition="roman_numeral",
        question="What is the primary treatment for streptococcal pharyngitis?",
        option_labels=["I", "II", "III", "IV"],
        option_texts=["Amoxicillin", "Vancomycin", "Ciprofloxacin", "Acyclovir"],
        options_displayed=True,
        gold_label="I",
        gold_text="Amoxicillin",
        gold_index=0,
    )


@pytest.fixture
def sample_fixed_pos_example():
    # Gold shifted to position D (index 3)
    return Example(
        question_id="test::001",
        source_dataset="test",
        source_id="001",
        condition="fixed_pos",
        question="What is the primary treatment for streptococcal pharyngitis?",
        option_labels=["A", "B", "C", "D"],
        option_texts=["Vancomycin", "Ciprofloxacin", "Acyclovir", "Amoxicillin"],
        options_displayed=True,
        gold_label="D",
        gold_text="Amoxicillin",
        gold_index=3,
    )


@pytest.fixture
def sample_no_symbols_example():
    return Example(
        question_id="test::001",
        source_dataset="test",
        source_id="001",
        condition="no_symbols",
        question="What is the primary treatment for streptococcal pharyngitis?",
        option_labels=[],
        option_texts=["Amoxicillin", "Vancomycin", "Ciprofloxacin", "Acyclovir"],
        options_displayed=True,
        gold_label=None,
        gold_text="Amoxicillin",
        gold_index=0,
    )


@pytest.fixture
def sample_nota_example():
    # Gold option replaced by "None of the provided options" at position A
    return Example(
        question_id="test::001",
        source_dataset="test",
        source_id="001",
        condition="none_of_the_provided",
        question="What is the primary treatment for streptococcal pharyngitis?",
        option_labels=["A", "B", "C", "D"],
        option_texts=["None of the provided options", "Vancomycin", "Ciprofloxacin", "Acyclovir"],
        options_displayed=True,
        gold_label="A",
        gold_text="None of the provided options",
        gold_index=0,
    )


@pytest.fixture
def sample_incorrect_example():
    # Model must select all incorrect options: B, C, D
    return Example(
        question_id="test::001",
        source_dataset="test",
        source_id="001",
        condition="incorrect",
        question="What is the primary treatment for streptococcal pharyngitis?",
        option_labels=["A", "B", "C", "D"],
        option_texts=["Amoxicillin", "Vancomycin", "Ciprofloxacin", "Acyclovir"],
        options_displayed=True,
        gold_label=["B", "C", "D"],
        gold_text=None,
        gold_index=0,  # original gold option was A (index 0)
        gold_indices=[1, 2, 3],
    )


def test_format_reward():
    assert compute_r_format('{"answer": "A"}') == 1.0
    assert compute_r_format('```json\n{"answer": "B"}\n```') == 0.8
    assert compute_r_format("Just the answer is A") == 0.0
    assert compute_r_format("") == 0.0


def test_mcq_rewards(sample_mcq_example):
    # Correct
    r_corr, pr, _ = compute_r_correct(sample_mcq_example, '{"answer": "A"}')
    r_symb, _ = compute_r_symbolic(sample_mcq_example, '{"answer": "A"}')
    assert r_corr == 1.0
    assert r_symb == 1.0

    # Incorrect valid option
    r_corr, pr, _ = compute_r_correct(sample_mcq_example, '{"answer": "B"}')
    r_symb, _ = compute_r_symbolic(sample_mcq_example, '{"answer": "B"}')
    assert r_corr == 0.0
    assert r_symb == 0.5  # valid label format, incorrect answer

    # Invalid / unparseable
    r_corr, pr, _ = compute_r_correct(sample_mcq_example, 'invalid text')
    r_symb, _ = compute_r_symbolic(sample_mcq_example, 'invalid text')
    assert r_corr == 0.0
    assert r_symb == 0.0


def test_roman_numeral_rewards(sample_roman_example):
    # Correct roman numeral
    r_corr, _, _ = compute_r_correct(sample_roman_example, '{"answer": "I"}')
    r_symb, _ = compute_r_symbolic(sample_roman_example, '{"answer": "I"}')
    assert r_corr == 1.0
    assert r_symb == 1.0

    # Wrong roman numeral
    r_corr, _, _ = compute_r_correct(sample_roman_example, '{"answer": "II"}')
    r_symb, _ = compute_r_symbolic(sample_roman_example, '{"answer": "II"}')
    assert r_corr == 0.0
    assert r_symb == 0.25  # valid format, wrong semantic answer

    # Letter leak (violates roman interface)
    r_corr, _, _ = compute_r_correct(sample_roman_example, '{"answer": "A"}')
    r_symb, _ = compute_r_symbolic(sample_roman_example, '{"answer": "A"}')
    assert r_corr == 0.0
    assert r_symb == 0.0  # interface violation


def test_fixed_pos_rewards(sample_fixed_pos_example):
    # Correct (shifted to D)
    r_corr, _, _ = compute_r_correct(sample_fixed_pos_example, '{"answer": "D"}')
    r_symb, _ = compute_r_symbolic(sample_fixed_pos_example, '{"answer": "D"}')
    assert r_corr == 1.0
    assert r_symb == 1.0

    # Position bias failure (picked old position A)
    r_corr, _, _ = compute_r_correct(sample_fixed_pos_example, '{"answer": "A"}')
    r_symb, _ = compute_r_symbolic(sample_fixed_pos_example, '{"answer": "A"}')
    assert r_corr == 0.0
    assert r_symb == 0.25


def test_no_symbols_rewards(sample_no_symbols_example):
    # Correct verbatim text
    r_corr, _, _ = compute_r_correct(sample_no_symbols_example, '{"answer": "Amoxicillin"}')
    r_symb, _ = compute_r_symbolic(sample_no_symbols_example, '{"answer": "Amoxicillin"}')
    assert r_corr == 1.0
    assert r_symb == 1.0

    # Distractor text
    r_corr, _, _ = compute_r_correct(sample_no_symbols_example, '{"answer": "Vancomycin"}')
    r_symb, _ = compute_r_symbolic(sample_no_symbols_example, '{"answer": "Vancomycin"}')
    assert r_corr == 0.0
    assert r_symb == 0.25

    # Letter label leak (violates no_symbols format)
    r_corr, _, _ = compute_r_correct(sample_no_symbols_example, '{"answer": "A"}')
    r_symb, _ = compute_r_symbolic(sample_no_symbols_example, '{"answer": "A"}')
    assert r_corr == 0.0
    assert r_symb == 0.0


def test_nota_adaptation_rewards(sample_nota_example):
    # Adapted correctly: selected NOTA
    r_corr, _, _ = compute_r_correct(sample_nota_example, '{"answer": "A"}')
    r_symb, _ = compute_r_symbolic(sample_nota_example, '{"answer": "A"}')
    assert r_corr == 1.0
    assert r_symb == 1.0

    # Adaptation failure: picked distractor B
    r_corr, _, _ = compute_r_correct(sample_nota_example, '{"answer": "B"}')
    r_symb, _ = compute_r_symbolic(sample_nota_example, '{"answer": "B"}')
    assert r_corr == 0.0
    assert r_symb == 0.0  # Required adaptation failure gives 0.0 symbolic reward!


def test_incorrect_task_inversion_rewards(sample_incorrect_example):
    # Full inversion success: selected all incorrect options [B, C, D]
    r_corr, _, _ = compute_r_correct(sample_incorrect_example, '{"answer": ["B", "C", "D"]}')
    r_symb, _ = compute_r_symbolic(sample_incorrect_example, '{"answer": ["B", "C", "D"]}')
    assert r_corr == 1.0
    assert r_symb == 1.0

    # Inversion format obeyed (3 options selected), but wrong complement [A, C, D]
    r_corr, _, _ = compute_r_correct(sample_incorrect_example, '{"answer": ["A", "C", "D"]}')
    r_symb, _ = compute_r_symbolic(sample_incorrect_example, '{"answer": ["A", "C", "D"]}')
    assert r_corr == 0.0
    assert r_symb == 0.5  # Syntax followed, medical knowledge wrong

    # Task inversion failure: emitted single option "A"
    r_corr, _, _ = compute_r_correct(sample_incorrect_example, '{"answer": "A"}')
    r_symb, _ = compute_r_symbolic(sample_incorrect_example, '{"answer": "A"}')
    assert r_corr == 0.0
    assert r_symb == 0.0  # Failed inversion structure


def test_combined_reward_lambda_scaling(sample_roman_example):
    # Perfect completion
    b_1 = compute_reward(sample_roman_example, '{"answer": "I"}', lambda_symbolic=1.0)
    assert b_1.r_correct == 1.0
    assert b_1.r_symbolic == 1.0
    assert b_1.r_total == 2.0

    b_2 = compute_reward(sample_roman_example, '{"answer": "I"}', lambda_symbolic=2.0)
    assert b_2.r_total == 3.0

    b_0 = compute_reward(sample_roman_example, '{"answer": "I"}', lambda_symbolic=0.0)
    assert b_0.r_total == 1.0  # Correctness only

    # Partially correct (format followed, wrong answer)
    b_part = compute_reward(sample_roman_example, '{"answer": "II"}', lambda_symbolic=1.0)
    assert b_part.r_correct == 0.0
    assert b_part.r_symbolic == 0.25
    assert b_part.r_total == 0.25


def test_batch_reward_functions(sample_mcq_example, sample_nota_example):
    import json
    from medperturb.reward.batch import (
        create_reward_ns_func,
        create_reward_symbolic_func,
        extract_completion_text,
        reconstruct_example,
        reward_correct_func,
        reward_format_func,
    )

    # Test extract_completion_text
    assert extract_completion_text("plain string") == "plain string"
    assert extract_completion_text([{"role": "assistant", "content": "chat msg"}]) == "chat msg"

    # Test reconstruct_example
    ex_dict = {
        "question_id": sample_mcq_example.question_id,
        "source_dataset": sample_mcq_example.source_dataset,
        "source_id": sample_mcq_example.source_id,
        "condition": sample_mcq_example.condition,
        "question": sample_mcq_example.question,
        "option_labels": sample_mcq_example.option_labels,
        "option_texts": sample_mcq_example.option_texts,
        "options_displayed": sample_mcq_example.options_displayed,
        "gold_label": sample_mcq_example.gold_label,
        "gold_text": sample_mcq_example.gold_text,
        "gold_index": sample_mcq_example.gold_index,
    }
    reconstructed = reconstruct_example(json.dumps(ex_dict))
    assert reconstructed.question_id == sample_mcq_example.question_id

    # Test reward_correct_func in batch
    examples_json = [json.dumps(ex_dict), json.dumps({**ex_dict, "condition": "none_of_the_provided", "gold_label": "A"})]
    completions = ['{"answer": "A"}', '{"answer": "B"}']
    corr_scores = reward_correct_func(prompts=["", ""], completions=completions, example_json=examples_json)
    assert corr_scores == [1.0, 0.0]

    # Test create_reward_symbolic_func in batch
    symb_fn = create_reward_symbolic_func(lambda_symbolic=1.5)
    symb_scores = symb_fn(prompts=["", ""], completions=completions, example_json=examples_json)
    assert symb_scores[0] == 1.5  # 1.5 * 1.0
    assert symb_scores[1] == 0.0  # NOTA failure -> 0.0

    # Test format reward
    fmt_scores = reward_format_func(prompts=["", ""], completions=['{"answer": "A"}', 'not json'])
    assert fmt_scores == [1.0, 0.0]

    # Test create_reward_ns_func
    ns_fn = create_reward_ns_func(lambda_symbolic=1.0, beta_format=0.1)
    ns_scores = ns_fn(prompts=["", ""], completions=completions, example_json=examples_json)
    assert ns_scores[0] == pytest.approx(1.0 + 1.0 + 0.1)

