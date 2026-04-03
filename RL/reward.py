"""
Reward functions for GRPO training.
Shared by both the GRPO loop (grpo_train.py) and can be used standalone.
"""

import ast
import numpy as np
from typing import Optional


def _has_syntax_error(code: str) -> bool:
    """Return True if code has a Python syntax error."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True


def _has_cot_reasoning(completion: str) -> bool:
    """Return True if the completion contains a <think> block before the code."""
    return "<think>" in completion and "</think>" in completion


def compute_rewards(
    codes: list,
    scores: list,
    completions: list,
    branch_manager,
    global_best: float,
) -> list:
    """
    Score a GRPO group of completions. Returns z-score normalized rewards.

    Args:
        codes: Extracted Python code strings (one per completion).
        scores: Execution times in ms, or float('inf') for failures.
        completions: Raw model outputs (used for CoT detection).
        branch_manager: BranchManager instance for novelty detection.
        global_best: Best score seen so far across all iterations.

    Returns:
        List of normalized reward floats (z-scored within the group).
    """
    raw_rewards = []
    for code, score, completion in zip(codes, scores, completions):
        r = _single_reward(code, score, completion, branch_manager, global_best)
        raw_rewards.append(r)

    # Z-score normalize within group (core GRPO insight)
    rewards = np.array(raw_rewards, dtype=np.float64)
    if rewards.std() > 1e-8:
        rewards = (rewards - rewards.mean()) / rewards.std()
    return rewards.tolist()


def _single_reward(
    code: str,
    score: float,
    completion: str,
    branch_manager,
    global_best: float,
) -> float:
    """Compute raw (un-normalized) reward for a single completion."""
    # --- Failure penalties (graduated, not flat -100) ---
    if score == float('inf') or (hasattr(score, '__float__') and np.isinf(score)):
        if not code.strip():
            return -5.0  # Empty generation
        elif _has_syntax_error(code):
            return -3.0  # Syntax error
        else:
            return -2.0  # Wrong output or timeout

    # --- Success reward ---
    # Base: inverse of runtime (lower ms = higher reward)
    r = 1000.0 / (score + 1.0)

    # Improvement bonus
    if score < global_best:
        r += 10.0

    # Exploration bonus for novel approach (not a previously-seen fingerprint)
    if branch_manager is not None and not branch_manager.registry.is_pruned(code):
        r += 1.0

    # CoT bonus — reward reasoning before code
    if _has_cot_reasoning(completion):
        r += 0.5

    return r


def compute_single_reward(
    code: str,
    score: float,
    global_best: float,
    branch_manager=None,
) -> float:
    """
    Simplified single-completion reward (no normalization).
    Used by the prompt-guided baseline (main.py) if needed.
    """
    if score == float('inf') or np.isinf(score):
        return -100.0
    performance_reward = 1000.0 / (score + 1.0)
    improvement_bonus = 50.0 if score < global_best else 0.0
    exploitation_score = 0.0
    if branch_manager is not None:
        report = branch_manager.detector.detect(
            branch_manager.branches.get(branch_manager.current_branch, [])
        )
        exploitation_score = report.get("exploitation_score", 0.0)
    exploitation_penalty = exploitation_score * 25.0
    reward = performance_reward + improvement_bonus - exploitation_penalty
    return float(reward) if not np.isnan(reward) else -100.0
