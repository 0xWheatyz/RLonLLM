# RLonLLM

Reinforcement learning fine-tuning of small edge language models (0.5B–1.5B params) on a focused coding task: generating fast prime-finding algorithms. The core hypothesis is that RL with exploitation detection can drive genuine algorithmic exploration rather than incremental micro-optimization.

## How It Works

Each training iteration, the model generates candidate `find_primes(n)` implementations. A multi-faceted reward function evaluates them on correctness and speed. An AST-based exploitation detector monitors whether the model is stuck refining one approach — if so, that branch is pruned and the model is steered toward novel algorithms.

**Training modes:**
- **GRPO** (Group Relative Policy Optimization) — Updates LoRA adapters using policy gradients. No GPU required.
- **Prompt-guided baseline** — Prompt-only exploration via Ollama (no weight updates).

**Classifier autoresearch** — A Junie agent loop automatically tunes exploitation detection thresholds by running live evals and committing changes.

## Project Structure

```
RL/
  grpo_train.py       # GRPO training loop with LoRA on CPU
  main.py             # Prompt-guided baseline (Ollama)
  classifer.py        # AST fingerprinter + exploitation detector
  reward.py           # Reward functions (correctness, speed, exploration bonus)
  model_configs.py    # Model registry and GRPO hyperparameters
  requirements.txt
  Dockerfile

autoresearch/
  run.sh              # Junie autoresearch loop
  program.md          # Instructions for the Junie agent
  eval.py             # Frozen evaluation harness (do not modify)

results/              # Training outputs per model
visualize_results.py  # Multi-model comparison plots
docker-compose.yml
mise.toml             # Task runner
```

## Quick Start

### Local (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r RL/requirements.txt

# Run GRPO training
cd RL
python grpo_train.py --model qwen-1.5b --iterations 30 --group-size 4
```

Results are written to `results/qwen-1.5b/`:
- `training_log.jsonl` — per-attempt metrics
- `summary.json` — final stats and best code
- `checkpoint-*/` — LoRA checkpoints per iteration

### Docker

```bash
# Start Ollama service
docker compose up ollama -d

# Train Qwen-1.5B
docker compose run --build rl-trainer python grpo_train.py --model qwen-1.5b
```

### mise tasks

```bash
mise run grpo-qwen       # GRPO training for Qwen2.5-Coder-1.5B
mise run grpo-deepseek   # GRPO training for DeepSeek-Coder-1.3B
mise run grpo-all        # Sequential training for both models
mise run autoresearch    # Launch Junie classifier tuning loop
mise run compare         # Visualize and compare results
mise run prefetch        # Pre-download HuggingFace models
```

## Supported Models

Configured in `RL/model_configs.py`:

| Key | Model |
|-----|-------|
| `qwen-1.5b` | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| `deepseek-1.3b` | deepseek-ai/deepseek-coder-1.3b-instruct |

## Key Components

### Exploitation Detector (`RL/classifer.py`)

Prevents the model from endlessly micro-optimizing one algorithm. Combines:

1. **Structural fingerprinting** — AST analysis identifies algorithmic patterns (Sieve variants, Trial Division, Miller-Rabin, Wheel Factorization, etc.) that are stable across variable renames.
2. **Reward delta trend** — Signals flattening when improvement per step falls below `reward_delta_threshold` (default 2%).
3. **Cosine similarity** — Flags convergence when recent attempts are structurally too similar (`similarity_threshold` default 0.92).

When exploitation is detected, the branch is pruned and the BranchManager injects a constraint into the next prompt:

```
EXPLICIT CONSTRAINT: Do not use any of these approaches: trial_division_basic, sieve_skip_even
```

### Reward Function (`RL/reward.py`)

| Event | Reward |
|-------|--------|
| Empty generation | −5 |
| Syntax error | −3 |
| Wrong output / timeout | −2 |
| Correct + fast | `1000 / (ms + 1)` |
| Beats global best | +10 |
| Novel fingerprint | +1 |
| CoT reasoning block | +0.5 |

Within each GRPO group, rewards are z-score normalized so relative performance drives the gradient, not absolute values.

### GRPO Training (`RL/grpo_train.py`)

Manual implementation (no `trl` library) to avoid hidden CUDA assumptions. For each iteration:

1. Generate K=4 completions from the current policy
2. Evaluate correctness (`find_primes(100)`) and speed (`find_primes(10^6)`)
3. Normalize rewards within the group
4. Policy gradient update: `−reward × log_prob + kl_coeff × KL_divergence`
5. Save LoRA checkpoint

LoRA config: rank=16, alpha=32, dropout=0.05, targets `q_proj`/`v_proj`.

### Autoresearch Loop (`autoresearch/`)

Junie autonomously tunes classifier thresholds:

```bash
cd autoresearch
bash run.sh 50   # up to 50 experiments
```

Each experiment:
1. Modifies one threshold in `RL/classifer.py`
2. Runs `eval.py` (10 iterations × 3 candidates)
3. Greps the composite score: `breadth×10 + 1000/best_ms − wasted×5`
4. Commits on improvement, reverts on regression
5. Logs to `results.tsv`

## System Requirements

- **RAM:** 64 GB (no GPU required; CPU-only fp32 training)
- **Storage:** ~50 GB (models + checkpoints)
- **Python:** 3.11+
- **Ollama:** Required for baseline and autoresearch eval

## Results

Best result from Qwen2.5-Coder-0.5B (30 iterations):

- **50.65 ms** for `find_primes(10^6)`
- Algorithm: Sieve of Eratosthenes with skip-even optimization
- 12 branches explored, 11 pruned by exploitation detector
