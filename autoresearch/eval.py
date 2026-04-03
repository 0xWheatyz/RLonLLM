"""
Frozen eval harness for classifier experiments.
DO NOT MODIFY — Junie optimizes RL/classifer.py against this.

Runs a shortened prompt-guided RL loop using Ollama (10 iterations x 3 candidates),
measures classifier effectiveness, outputs grepable metrics.

Output format (one metric per line, grep-friendly):
    eval_breadth:     <int>
    eval_best_ms:     <float>
    eval_wasted:      <int>
    eval_score:       <float>
    eval_status:      ok|fail
"""

import os
import sys
import re
import json
import time
import subprocess
import requests
import numpy as np
import random

# Fixed seed for reproducibility — do not change
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configuration — frozen
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("EVAL_MODEL", "qwen2.5-coder:1.5b")
TARGET_N = 10**6
EVAL_ITERATIONS = 10
GENERATION_BATCH_SIZE = 3

# Import the classifier from RL/ — path adjustment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RL'))
from classifer import BranchManager, Attempt


def generate_code(prompt: str) -> str:
    """Call Ollama to generate a find_primes implementation."""
    url = f"{OLLAMA_HOST}/api/generate"
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "seed": RANDOM_SEED},
    }
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        raw = response.json().get("response", "")
        match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
        return match.group(1).strip() if match else raw.strip()
    except Exception as e:
        print(f"[eval] generation error: {e}", file=sys.stderr)
        return ""


def evaluate_code(code: str) -> float:
    """Run code in subprocess, return ms or float('inf') on failure."""
    if not code:
        return float('inf')
    test_script = f"""
{code}

def check():
    try:
        primes = find_primes(100)
        expected = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
        return list(primes) == expected
    except Exception:
        return False

if __name__ == "__main__":
    import time
    if check():
        start = time.perf_counter()
        find_primes({TARGET_N})
        end = time.perf_counter()
        print((end - start) * 1000)
    else:
        print("FAILED")
"""
    try:
        res = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True, text=True, timeout=15
        )
        if res.returncode != 0 or res.stdout.strip() == "FAILED":
            return float('inf')
        val = res.stdout.strip().split('\n')[-1]
        return float(val)
    except Exception:
        return float('inf')


def run_eval() -> dict:
    """Run the full evaluation and return metrics dict."""
    branch_manager = BranchManager(
        exploit_threshold=0.75,
        reward_delta_threshold=0.02,
        window=4,
    )

    best_ms = float('inf')
    distinct_approaches = set()
    wasted_count = 0  # branches pruned that contained global best at time of pruning
    global_best_at_prune = float('inf')

    system_prompt = (
        "You are an expert Python optimizer. Write a function `find_primes(n)` "
        "that returns a list of all primes up to n. Maximize speed. "
        "Output only a ```python code block. No explanations."
    )

    for i in range(EVAL_ITERATIONS):
        history_context = branch_manager.get_prompt_context()
        prompt = f"{system_prompt}\n\n{history_context}\n\nWrite find_primes(n)."

        for j in range(GENERATION_BATCH_SIZE):
            code = generate_code(prompt)
            score = evaluate_code(code)
            report = branch_manager.record(code, score)

            label = report.get("approach_label", "unknown")
            if score != float('inf'):
                distinct_approaches.add(label)
                if score < best_ms:
                    best_ms = score

            if report.get("action") == "PRUNE_AND_BRANCH":
                # Check if we're pruning a branch that had the current best
                branch_attempts = branch_manager.branches.get(branch_manager.current_branch, [])
                branch_best = min((a.score for a in branch_attempts if a.score != float('inf')), default=float('inf'))
                if branch_best <= global_best_at_prune:
                    wasted_count += 1
                global_best_at_prune = min(global_best_at_prune, best_ms)
                break

    if best_ms == float('inf'):
        return {"status": "fail", "breadth": 0, "best_ms": 9999.0, "wasted": wasted_count}

    return {
        "status": "ok",
        "breadth": len(distinct_approaches),
        "best_ms": best_ms,
        "wasted": wasted_count,
    }


def compute_score(breadth: int, best_ms: float, wasted: int) -> float:
    """Composite score. Higher is better."""
    return (breadth * 10) + (1000.0 / best_ms) - (wasted * 5)


if __name__ == "__main__":
    try:
        metrics = run_eval()
        status = metrics["status"]
        breadth = metrics["breadth"]
        best_ms = metrics["best_ms"]
        wasted = metrics["wasted"]
        score = compute_score(breadth, best_ms, wasted) if status == "ok" else 0.0

        print(f"eval_breadth:     {breadth}")
        print(f"eval_best_ms:     {best_ms:.4f}")
        print(f"eval_wasted:      {wasted}")
        print(f"eval_score:       {score:.4f}")
        print(f"eval_status:      {status}")
        sys.exit(0)
    except Exception as e:
        print(f"eval_breadth:     0", flush=True)
        print(f"eval_best_ms:     9999.0000", flush=True)
        print(f"eval_wasted:      0", flush=True)
        print(f"eval_score:       0.0000", flush=True)
        print(f"eval_status:      fail", flush=True)
        print(f"[eval] fatal error: {e}", file=sys.stderr)
        sys.exit(1)
