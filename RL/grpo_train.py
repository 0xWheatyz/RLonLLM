"""
GRPO Training Script for find_primes RL
========================================
Manual GRPO loop (no trl.GRPOTrainer) — avoids hidden CUDA assumptions.
Trains LoRA adapters using HuggingFace transformers + peft.
Supports GPU (CUDA) with automatic fallback to CPU.

Usage:
    python grpo_train.py --model qwen-1.5b
    python grpo_train.py --model deepseek-1.3b --iterations 15 --group-size 2
"""

import argparse
import datetime
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from classifer import BranchManager, Attempt
from model_configs import MODELS, GRPO_CONFIG
from reward import compute_rewards


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_N = 10**6

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results")

SYSTEM_PROMPT = (
    "You are an expert Python optimizer. Your task is to write a function "
    "`find_primes(n)` that returns a list of all primes up to n (inclusive). "
    "The goal is maximum speed.\n\n"
    "You MUST structure your response exactly as follows:\n"
    "<think>\n... your reasoning about the approach ...\n</think>\n"
    "```python\n... your code ...\n```\n\n"
    "Consider optimizations like: Sieve of Eratosthenes, skipping even numbers, "
    "segmented sieves for memory efficiency, or bit manipulation with bytearray. "
    "DO NOT use external libraries like numpy or sympy."
)


# ---------------------------------------------------------------------------
# Code evaluation (standalone — mirrors main.py logic)
# ---------------------------------------------------------------------------

def evaluate_code(code: str) -> float:
    """
    Execute generated code in a subprocess, verify correctness against
    find_primes(100), then benchmark find_primes(10**6).
    Returns runtime in milliseconds, or float('inf') on any failure.
    """
    if not code:
        return float("inf")

    test_script = f"""
{code}

def check():
    try:
        primes = find_primes(100)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if list(primes) != expected:
            import sys
            if 2 not in primes:
                print("FAILED: Missing prime 2", file=sys.stderr)
            elif len(primes) != len(expected):
                print(f"FAILED: Wrong count. Got {{len(primes)}}, expected {{len(expected)}}", file=sys.stderr)
            else:
                print("FAILED: Primes do not match expected list", file=sys.stderr)
            return False
        return True
    except Exception as e:
        import sys
        print(f"FAILED: {{e}}", file=sys.stderr)
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
        import resource
        def _limit_resources():
            # 512MB virtual memory limit
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            # No subprocess spawning (prevents fork bombs)
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

        res = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            preexec_fn=_limit_resources,
        )
        if res.returncode != 0:
            print(f"  [eval] exit {res.returncode}: {res.stderr.strip()}")
            return float("inf")

        output = res.stdout.strip()
        if output == "FAILED":
            print(f"  [eval] correctness check failed: {res.stderr.strip()}")
            return float("inf")

        val = output.split("\n")[-1]
        return float(val)

    except subprocess.TimeoutExpired:
        return float("inf")
    except Exception as e:
        print(f"  [eval] error: {e}")
        return float("inf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_python_code(text: str) -> str:
    """Extract the first ```python ... ``` block from model output."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def has_cot(text: str) -> bool:
    """Check whether the completion contains a <think>...</think> block."""
    return "<think>" in text and "</think>" in text


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def compute_log_probs(model, input_ids: torch.Tensor, completion_start: int, device: torch.device = None) -> torch.Tensor:
    """
    Compute the sum of per-token log-probabilities for the *completion* portion
    of `input_ids`.

    Args:
        model: the language model (policy or reference).
        input_ids: (1, seq_len) token ids for prompt + completion.
        completion_start: index where the completion tokens begin.
        device: target device for tensors.

    Returns:
        Scalar tensor — sum of log-probs over the completion tokens.
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(input_ids)
    # logits shape: (1, seq_len, vocab_size)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    # For each completion token position t, the relevant logit is at position t-1
    # (the model predicts token t from position t-1).
    completion_ids = input_ids[0, completion_start:]  # (comp_len,)
    # Corresponding logits are at positions (completion_start-1) .. (seq_len-2)
    pred_log_probs = log_probs[0, completion_start - 1 : -1, :]  # (comp_len, vocab)

    # Gather the log-prob of each actual token
    token_log_probs = pred_log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(model_key: str):
    """
    Load the tokenizer, a LoRA-wrapped policy model, and a frozen reference
    model.  Uses dtype from GRPO_CONFIG and moves models to GPU if available.

    Returns:
        (tokenizer, policy_model, ref_model, model_config, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = DTYPE_MAP.get(GRPO_CONFIG.get("dtype", "float32"), torch.float32)

    model_cfg = MODELS[model_key]
    hf_id = model_cfg["hf_id"]
    print(f"Loading {hf_id} ...")
    print(f"  Device: {device}  |  Dtype: {model_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Policy model (LoRA) ---
    base_model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    base_model.to(device)
    lora_cfg = LoraConfig(
        r=GRPO_CONFIG["lora_rank"],
        lora_alpha=GRPO_CONFIG["lora_alpha"],
        lora_dropout=GRPO_CONFIG["lora_dropout"],
        target_modules=model_cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(base_model, lora_cfg)
    policy_model.train()

    # Enable gradient checkpointing if configured (saves VRAM)
    if GRPO_CONFIG.get("gradient_checkpointing"):
        policy_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    policy_model.print_trainable_parameters()

    # --- Reference model (frozen, no LoRA) ---
    ref_model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return tokenizer, policy_model, ref_model, model_cfg, device


# ---------------------------------------------------------------------------
# Single generation
# ---------------------------------------------------------------------------

def generate_completion(
    policy_model,
    tokenizer,
    prompt_text: str,
    temperature: float,
    max_new_tokens: int,
    device: torch.device = None,
) -> tuple[str, torch.Tensor, int]:
    """
    Generate a single completion from the policy model.

    Returns:
        (decoded_completion, full_input_ids, prompt_length)
        full_input_ids is (1, prompt_len + completion_len).
    """
    if device is None:
        device = next(policy_model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_length = prompt_ids.shape[1]

    with torch.no_grad():
        output_ids = policy_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion_ids = output_ids[0, prompt_length:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_text, output_ids, prompt_length


# ---------------------------------------------------------------------------
# GRPO update step
# ---------------------------------------------------------------------------

def grpo_update(
    policy_model,
    ref_model,
    optimizer,
    completions_data: list[dict],
    normalized_rewards: list[float],
    kl_coeff: float,
    device: torch.device = None,
) -> float:
    """
    Perform one GRPO policy gradient step over a group of completions.

    Each entry in completions_data has keys:
        "input_ids": (1, seq_len) tensor
        "prompt_length": int

    Returns:
        The scalar loss value.
    """
    if device is None:
        device = next(policy_model.parameters()).device

    policy_model.train()
    optimizer.zero_grad()
    total_loss_value = 0.0
    n_valid = 0

    for comp, reward in zip(completions_data, normalized_rewards):
        input_ids = comp["input_ids"].to(device)
        prompt_len = comp["prompt_length"]

        if input_ids.shape[1] <= prompt_len:
            # No completion tokens — skip
            continue

        # Policy log-probs (need gradients here)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = policy_model(input_ids)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        completion_ids = input_ids[0, prompt_len:]
        pred_log_probs = log_probs[0, prompt_len - 1 : -1, :]
        token_lp = pred_log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
        policy_lp = token_lp.sum()

        # Reference log-probs (no gradient)
        ref_lp = compute_log_probs(ref_model, input_ids, prompt_len, device=device)

        # KL divergence for this completion: sum(policy_lp - ref_lp)
        kl = policy_lp - ref_lp

        # GRPO loss: -reward * policy_log_prob + kl_coeff * KL
        # Divide by total count now so gradients accumulate correctly
        sample_loss = (-reward * policy_lp + kl_coeff * kl) / len(completions_data)
        sample_loss.backward()
        total_loss_value += sample_loss.item()
        n_valid += 1

    if n_valid > 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

    return total_loss_value


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seeds(args.seed)

    # Resolve model
    if args.model not in MODELS:
        print(f"Unknown model '{args.model}'. Available: {list(MODELS.keys())}")
        sys.exit(1)

    tokenizer, policy_model, ref_model, model_cfg, device = load_models(args.model)

    optimizer = AdamW(
        policy_model.parameters(),
        lr=GRPO_CONFIG["learning_rate"],
        weight_decay=GRPO_CONFIG.get("weight_decay", 0.01),
    )

    # Directories
    results_dir = os.path.join(RESULTS_BASE, args.model)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_log.jsonl")

    branch_manager = BranchManager(
        exploit_threshold=0.75,
        reward_delta_threshold=0.02,
        window=4,
    )

    best_score = float("inf")
    best_code = ""
    start_iteration = 1

    # Resume from previous run if log exists and --resume is set
    if args.resume and os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                start_iteration = max(start_iteration, entry["iteration"] + 1)
                if entry["score"] < best_score:
                    best_score = entry["score"]
                    best_code = entry.get("code", "")
                # Replay into branch_manager so exploitation state is restored
                branch_manager.record(entry.get("code", ""), entry["score"])
        print(f"Resuming from iteration {start_iteration} (best so far: {best_score:.2f}ms)")
        # Load the latest checkpoint into the policy model
        last_ckpt = os.path.join(results_dir, f"checkpoint-{start_iteration - 1}")
        if os.path.isdir(last_ckpt):
            from peft import PeftModel
            policy_model = PeftModel.from_pretrained(
                policy_model.base_model.model, last_ckpt
            ).to(device)
            policy_model.train()
            print(f"  Loaded checkpoint: {last_ckpt}")
    else:
        # Fresh run — clear log
        with open(log_path, "w") as f:
            pass

    temperature = GRPO_CONFIG.get("temperature", 0.7)
    max_new_tokens = GRPO_CONFIG.get("max_new_tokens", 1024)
    kl_coeff = GRPO_CONFIG.get("kl_coeff", 0.05)

    print(f"\n{'='*60}")
    print(f"GRPO Training: {args.model}")
    print(f"Iterations: {start_iteration}-{args.iterations}  |  Group size: {args.group_size}")
    print(f"Temperature: {temperature}  |  KL coeff: {kl_coeff}")
    print(f"Results: {results_dir}")
    print(f"{'='*60}\n")

    for iteration in range(start_iteration, args.iterations + 1):
        print(f"\n--- Iteration {iteration}/{args.iterations} ---")

        # Build prompt with approach history
        history_context = branch_manager.get_prompt_context()
        prompt_text = f"{SYSTEM_PROMPT}\n\n{history_context}\n\nWrite a highly optimized find_primes(n) function."

        # Generate group_size completions
        group_completions = []  # list of dicts
        group_codes = []
        group_scores = []

        for g in range(args.group_size):
            print(f"  Generating {g+1}/{args.group_size} ...", end="", flush=True)
            try:
                completion_text, full_ids, prompt_len = generate_completion(
                    policy_model, tokenizer, prompt_text, temperature, max_new_tokens,
                    device=device,
                )
            except (MemoryError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f" OOM/error: {e}")
                group_codes.append("")
                group_scores.append(float("inf"))
                group_completions.append(None)
                continue

            code = extract_python_code(completion_text)
            score = evaluate_code(code)

            if score != float("inf"):
                print(f" score={score:.2f}ms")
            else:
                print(" FAILED")

            group_codes.append(code)
            group_scores.append(score)
            group_completions.append({
                "input_ids": full_ids,
                "prompt_length": prompt_len,
                "text": completion_text,
            })

        # Compute normalized rewards via reward.py
        group_texts = [c["text"] if c else "" for c in group_completions]
        normalized_rewards = compute_rewards(
            group_codes, group_scores, group_texts, branch_manager, best_score,
        )

        # GRPO policy gradient update (only on completions that generated tokens)
        valid_comps = []
        valid_rewards = []
        for comp, reward in zip(group_completions, normalized_rewards):
            if comp is not None:
                valid_comps.append(comp)
                valid_rewards.append(reward)

        if valid_comps:
            loss = grpo_update(
                policy_model, ref_model, optimizer,
                valid_comps, valid_rewards, kl_coeff,
                device=device,
            )
            print(f"  Loss: {loss:.6f}")
        else:
            loss = float("nan")
            print("  No valid completions — skipping update.")

        # Record results + update BranchManager
        for g in range(args.group_size):
            code = group_codes[g]
            score = group_scores[g]
            reward = normalized_rewards[g]

            report = branch_manager.record(code, score)

            if score < best_score:
                best_score = score
                best_code = code

            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "iteration": iteration,
                "batch_index": g + 1,
                "score": score,
                "reward": reward,
                "approach_label": report.get("approach_label"),
                "exploitation_score": report.get("exploitation_score", 0.0),
                "is_exploiting": report.get("exploiting", False),
                "action": report.get("action"),
                "loss": loss,
                "model": args.model,
                "cot_present": has_cot(group_completions[g]["text"]) if group_completions[g] else False,
                "code": code,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if report["action"] == "PRUNE_AND_BRANCH":
                print(f"  [PRUNE] {report['message']}")

        # Save LoRA checkpoint
        ckpt_dir = os.path.join(results_dir, f"checkpoint-{iteration}")
        policy_model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        # Iteration summary
        valid_scores = [s for s in group_scores if s != float("inf")]
        avg = np.mean(valid_scores) if valid_scores else float("inf")
        print(
            f"  Scores: {[round(s, 2) if s != float('inf') else 'inf' for s in group_scores]} "
            f"| Rewards: {[round(r, 3) for r in normalized_rewards]} "
            f"| Avg: {avg:.2f}ms | Best: {best_score:.2f}ms"
        )

        # Free GPU memory between iterations to prevent OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    print("Training complete.")
    print(branch_manager.status())
    print(f"Best score: {best_score:.2f}ms")

    summary = {
        "model": args.model,
        "iterations": args.iterations,
        "group_size": args.group_size,
        "seed": args.seed,
        "best_score": best_score,
        "best_code": best_code,
        "final_status": branch_manager.status(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for find_primes RL task"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help=f"Model key from model_configs.py (e.g. qwen-1.5b)",
    )
    parser.add_argument(
        "--iterations", type=int, default=30,
        help="Number of GRPO training iterations (default: 30)",
    )
    parser.add_argument(
        "--group-size", type=int, default=4,
        help="Number of completions per group (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint instead of starting fresh",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
