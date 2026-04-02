# Weekend Refactor Plan: GRPO Training Pipeline + Junie Autoresearch Loop

**Created:** 2026-04-02
**Status:** Reviewed (Critic pass applied)
**Scope:** 2 workstreams, weekend timeline
**Review notes:** Plan reviewed by Critic agent. 6 issues found and fixed (see changelog at bottom).

---

## Requirements Summary

### Workstream 1: GRPO RL Fine-Tuning Pipeline
- Replace the current prompt-guided search (`RL/main.py`) with actual GRPO weight updates
- Train with LoRA on CPU (64GB RAM, no usable GPU — AMD 5500XT has no ROCm support)
- Test 2 models: Qwen2.5-Coder-1.5B, DeepSeek-Coder-1.3B (Phi-3.5-mini is 3.8B params — stretch goal only)
- Keep the existing eval harness (correctness at n=100, benchmark at n=10^6)
- Force CoT (reasoning before code) per paper finding that it's load-bearing for RL
- Reward normalization to prevent failure penalty (-100) from dominating

### Workstream 2: Junie Autoresearch Loop for Classifier
- Junie CLI (`junie "<prompt>"`) iterates on `classifer.py` thresholds
- Live RL runs as eval metric (not replay)
- Git commit/reset cycle (keep improvements, revert regressions)
- Start with numeric thresholds only; expand to known patterns if gains plateau
- Frozen eval harness with grepable output

---

## Architecture Decision: All-HuggingFace (No Ollama for Training)

**Decision:** Drop Ollama for the GRPO pipeline. Use HuggingFace `transformers` + `peft` with a manual GRPO loop. Keep Ollama running for the Junie autoresearch eval harness (Workstream 2).

**Why:**
- Ollama has no training API — you'd need a complex weight-sync pipeline (LoRA merge → GGUF convert → Ollama reload) that's fragile and slow
- CPU generation for 1.5B is ~10-30s per sample, which is acceptable given "time isn't a concern"
- Single framework = fewer moving parts for a weekend build
- Manual GRPO loop (~50 lines of REINFORCE math) is simpler and more transparent than `trl.GRPOTrainer`, which has hidden CUDA assumptions

**Tradeoff:** Generation is ~5-10x slower than Ollama's optimized GGUF inference. Acceptable for this project.

**Note:** Ollama must remain running for Workstream 2 — the Junie eval harness uses the existing prompt-guided loop via Ollama for fast iteration.

---

## Project Structure (After Refactor)

```
/home/wlee/.home/
├── RL/
│   ├── main.py                # KEEP AS-IS — baseline prompt-guided loop
│   ├── classifer.py           # KEEP — Junie's target file
│   ├── grpo_train.py          # NEW — GRPO training entry point
│   ├── reward.py              # NEW — reward functions (shared by both loops)
│   ├── model_configs.py       # NEW — LoRA/model configs for all 3 models
│   ├── requirements.txt       # UPDATED — add transformers, trl, peft, etc.
│   └── Dockerfile             # UPDATED — heavier deps
├── autoresearch/              # NEW — Junie loop
│   ├── program.md             # Instructions for Junie
│   ├── run.sh                 # Main loop script
│   ├── eval.py                # Frozen eval harness
│   └── results.tsv            # Experiment log (gitignored)
├── docker-compose.yml         # UPDATED — add grpo-trainer service
├── mise.toml                  # UPDATED — new tasks
├── results/                   # Training outputs (per model)
│   ├── qwen-1.5b/
│   ├── deepseek-1.3b/
│   └── phi-3.5/
└── visualize_results.py       # UPDATED — compare models
```

---

## Workstream 1: GRPO Training Pipeline

### Step 1: Dependencies and Config (`RL/requirements.txt`, `RL/model_configs.py`)

**File: `RL/requirements.txt`** — Add:
```
torch>=2.1.0                # CPU-only build (no CUDA)
transformers>=4.40.0
peft>=0.13.0                # LoRA
accelerate>=0.30.0          # Device placement
numpy
pandas                      # For visualize_results.py
matplotlib                  # For visualize_results.py
requests                    # Keep for Ollama baseline comparison
astunparse
```

**File: `RL/model_configs.py`** — Model registry:
```python
MODELS = {
    "qwen-1.5b": {
        "hf_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "lora_target_modules": ["q_proj", "v_proj"],
        "chat_template": "qwen",  # <|im_start|> format
    },
    "deepseek-1.3b": {
        "hf_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "lora_target_modules": ["q_proj", "v_proj"],
        "chat_template": "deepseek",
    },
}

# Stretch goal — Phi-3.5-mini is 3.8B params (not 1.5B), ~15GB fp32.
# Still fits in 64GB RAM but 2-3x slower training. Only attempt if
# Qwen + DeepSeek complete with time to spare.
STRETCH_MODELS = {
    "phi-3.5": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "lora_target_modules": ["qkv_proj"],  # Phi uses fused QKV
        "chat_template": "phi",
    },
}

# Shared GRPO config
GRPO_CONFIG = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "group_size": 4,          # GRPO generates N completions per prompt
    "max_new_tokens": 512,    # Paper: 512 tokens ≈ 2B→9B parameter jump
    "learning_rate": 1e-5,
    "kl_coeff": 0.05,         # KL penalty against reference policy
    "num_iterations": 30,
    "temperature": 0.7,
    "dtype": "float32",       # CPU — no bf16 guarantee
}
```

**Acceptance criteria:**
- [ ] `pip install -r requirements.txt` succeeds in a clean venv
- [ ] Each model in `MODELS` downloads and loads on CPU within 64GB RAM
- [ ] LoRA adapter attaches without error for each model

### Step 2: Reward Function (`RL/reward.py`)

Extract and improve the reward function from `main.py:142-172`. Key changes from the paper:

1. **Reward normalization** — z-score normalize within each GRPO group (the -100 failure penalty currently dominates)
2. **CoT bonus** — reward presence of reasoning tokens before code
3. **Validity shaping** — graduated penalties (syntax error: -5, wrong output: -3, timeout: -2) instead of flat -100
4. **Exploration bonus** — +1 for novel fingerprint (paper's "try-all-actions" equivalent)

```python
def compute_rewards(codes: list[str], scores: list[float], 
                    branch_manager, global_best: float) -> list[float]:
    """Score a GRPO group of completions. Returns normalized rewards."""
    raw_rewards = []
    for code, score in zip(codes, scores):
        if score == float('inf'):
            # Graduated failure penalty
            if not code.strip():
                r = -5.0  # Empty generation
            elif _has_syntax_error(code):
                r = -3.0  # Syntax error
            else:
                r = -2.0  # Wrong output or timeout
        else:
            # Performance reward (inverse of runtime)
            r = 1000.0 / (score + 1.0)
            # Improvement bonus
            if score < global_best:
                r += 10.0
            # Exploration bonus for novel approach
            report = branch_manager.record(code, score)
            if not branch_manager.registry.is_pruned(code):
                r += 1.0
        raw_rewards.append(r)
    
    # Z-score normalize within group (GRPO core insight)
    rewards = np.array(raw_rewards)
    if rewards.std() > 1e-8:
        rewards = (rewards - rewards.mean()) / rewards.std()
    return rewards.tolist()
```

**Acceptance criteria:**
- [ ] A group of [correct_fast, correct_slow, syntax_error, wrong_output] produces normalized rewards where correct_fast > correct_slow > wrong_output > syntax_error
- [ ] Normalization prevents any single penalty from being >2 std devs from mean

### Step 3: GRPO Training Script (`RL/grpo_train.py`)

This is the main new file. Implements a **manual GRPO loop** (~50 lines of REINFORCE math) rather than using `trl.GRPOTrainer` (which has hidden CUDA assumptions and is overkill for single-device CPU).

**Key design decisions:**
- **Manual GRPO over trl**: Generate N completions → score → z-score normalize rewards within group → compute log-prob weighted policy gradient → update LoRA weights via AdamW. Simple, transparent, no framework surprises on CPU.
- **CoT output format**: Force `<think>...</think>\n```python\n...\n``` ` structure. The paper shows CoT is load-bearing — without it, RL degrades to ICL-level performance.
- **GRPO group size = 4**: Generate 4 completions per prompt, score all 4, normalize rewards within the group, update policy. This is the DeepSeek-R1 approach — no value head needed.
- **CPU memory**: `torch_dtype=torch.float32`. Model (~3GB fp32) + LoRA (~50MB) + optimizer states (~100MB) + activations (~2-4GB) = ~6-8GB RAM for 1.5B models. Phi-3.5 (3.8B) would be ~15-20GB — stretch goal only.
- **Checkpoint per iteration**: Save LoRA adapter after each iteration so we can compare training progress across models.
- **Reproducibility**: Fixed random seed for torch, numpy, and Python hash seed.

**Pseudocode structure:**
```python
def main(model_name: str):
    # 1. Load model + LoRA
    config = MODELS[model_name]
    model = AutoModelForCausalLM.from_pretrained(config["hf_id"])
    model = get_peft_model(model, LoraConfig(...))
    
    # 2. Build prompt with CoT instruction
    system_prompt = """You are an expert Python optimizer. 
    First, reason about the algorithm in <think> tags.
    Then write the function in a ```python block.
    ..."""
    
    # 3. GRPO loop
    for iteration in range(NUM_ITERATIONS):
        # Generate group of completions
        completions = [generate(model, prompt) for _ in range(GROUP_SIZE)]
        
        # Extract code from each, evaluate
        codes = [extract_code(c) for c in completions]
        scores = [evaluate_code(c) for c in codes]
        
        # Compute normalized rewards
        rewards = compute_rewards(codes, scores, branch_manager, global_best)
        
        # GRPO update (policy gradient with group-normalized advantages)
        loss = grpo_update(model, ref_model, completions, rewards, kl_coeff)
        
        # Inject approach history into next prompt
        prompt = update_prompt(system_prompt, branch_manager)
        
        # Checkpoint
        model.save_pretrained(f"results/{model_name}/checkpoint-{iteration}")
```

**Acceptance criteria:**
- [ ] Script runs end-to-end on CPU for 1 iteration with Qwen-1.5B in <30 min
- [ ] LoRA weights checkpoint saves after each iteration (~2MB each)
- [ ] Training log JSONL is compatible with existing `visualize_results.py`
- [ ] CoT reasoning appears in >80% of generations after iteration 5

### Step 4: Multi-Model Runner and Comparison

**File: `mise.toml`** — Add tasks:
```toml
[tasks.grpo-qwen]
run = "cd RL && python grpo_train.py --model qwen-1.5b"

[tasks.grpo-deepseek]
run = "cd RL && python grpo_train.py --model deepseek-1.3b"

[tasks.grpo-phi]
run = "cd RL && python grpo_train.py --model phi-3.5"

[tasks.grpo-all]
description = "Run GRPO on all models sequentially"
run = """
cd RL && \
python grpo_train.py --model qwen-1.5b && \
python grpo_train.py --model deepseek-1.3b && \
python grpo_train.py --model phi-3.5
"""

[tasks.compare]
description = "Compare all model results"
run = "python visualize_results.py --compare"
```

**File: `visualize_results.py`** — Update to:
- Load results from `results/{model_name}/training_log.jsonl`
- Plot side-by-side: best runtime over iterations, algorithmic breadth, reward curves
- Output a comparison table: model | best_time_ms | approaches_found | iterations_to_best

**Acceptance criteria:**
- [ ] Each model produces a `training_log.jsonl` in its own results subdirectory
- [ ] `mise run compare` generates a comparison PNG with all 3 models
- [ ] Can identify which model discovered the most distinct approaches

### Step 5: Docker Update

Update `docker-compose.yml` to add a `grpo-trainer` service that doesn't depend on Ollama:

```yaml
grpo-trainer:
  build:
    context: .
    dockerfile: RL/Dockerfile
  environment:
    - MODEL_NAME=qwen-1.5b
    - RESULTS_DIR=/app/results
  volumes:
    - ./results:/app/results
    - huggingface_cache:/root/.cache/huggingface
  deploy:
    resources:
      limits:
        memory: 48G    # Leave headroom from 64GB
```

Keep the existing `ollama` + `rl-trainer` services as the baseline.

**Acceptance criteria:**
- [ ] `docker compose run grpo-trainer` completes 1 iteration
- [ ] HuggingFace model cache persists across runs (no re-download)

---

## Workstream 2: Junie Autoresearch Loop

### Step 6: Frozen Eval Harness (`autoresearch/eval.py`)

This file is **immutable** — Junie must never touch it. It runs a shortened RL loop and outputs grepable metrics.

```python
"""
Frozen eval harness for classifier experiments.
DO NOT MODIFY — Junie optimizes classifer.py against this.

Runs a 10-iteration prompt-guided RL loop using Ollama,
measures classifier effectiveness, outputs grepable metrics.
"""

# Uses the existing Ollama setup (not GRPO — faster iteration for Junie)
# Imports classifer.py to get BranchManager with current thresholds
# Runs 10 iterations × 3 candidates = 30 total attempts
# Outputs:
#   eval_breadth:     <int>    # distinct correct approaches found
#   eval_best_ms:     <float>  # fastest correct solution (ms)
#   eval_wasted:      <int>    # branches pruned that contained the best solution
#   eval_score:       <float>  # composite: breadth*10 + 1000/best_ms - wasted*5
#   eval_status:      ok|fail
```

**Composite score formula:**
```
eval_score = (eval_breadth * 10) + (1000 / eval_best_ms) - (eval_wasted * 5)
```

This rewards:
- Finding many distinct approaches (breadth)
- Finding fast solutions (best_ms)
- Not pruning winning branches too early (wasted)

**Acceptance criteria:**
- [ ] `python autoresearch/eval.py` outputs exactly the 5 grepable lines
- [ ] Running it twice with same `classifer.py` produces similar scores (±10%)
- [ ] Running with obviously bad thresholds (exploit_threshold=0.01) produces measurably worse score

### Step 7: Junie Program File (`autoresearch/program.md`)

```markdown
# Classifier Optimization Program

## Your Task
You are optimizing `RL/classifer.py` to improve algorithmic search effectiveness.
You may ONLY modify numeric thresholds and constants in `RL/classifer.py`.
You may NOT modify `autoresearch/eval.py` or any other file.

## The Metric
After each change, run: `python autoresearch/eval.py`
Extract: `grep "^eval_score:" run.log`
Higher eval_score = better. Current baseline is in results.tsv.

## What You Can Change (in classifer.py)
- ExploitationDetector.__init__: reward_delta_threshold, window, 
  similarity_threshold, min_attempts
- BranchManager.__init__: exploit_threshold, reward_delta_threshold, window
- PatternRegistry.__init__: similarity_threshold
- ExploitationDetector.detect: the 0.6/0.4 signal weights, 
  the 0.8/0.5 greedy zone thresholds
- cosine_similarity threshold in is_pruned()

## Experiment Loop
1. Read current `RL/classifer.py` and `autoresearch/results.tsv`
2. Form a hypothesis about which threshold change will improve eval_score
3. Edit `RL/classifer.py` with ONE threshold change
4. `git add RL/classifer.py && git commit -m "experiment: <description>"`
5. Run: `python autoresearch/eval.py > run.log 2>&1`
6. Extract: `grep "^eval_score:\|^eval_breadth:\|^eval_best_ms:" run.log`
7. If improved: log to results.tsv, keep the commit
8. If equal/worse: `git reset --hard HEAD~1`, log as "reverted"
9. REPEAT — do not stop or ask questions

## Rules
- ONE change per experiment (isolate variables)
- Log every experiment to results.tsv (hash, score, description, status)
- If 3 consecutive experiments show no improvement, try a DIFFERENT threshold
- Never modify eval.py
```

**Acceptance criteria:**
- [ ] `program.md` is clear enough that `junie` can follow it without clarification
- [ ] The experiment loop is self-contained (no human intervention needed)

### Step 8: Loop Runner (`autoresearch/run.sh`)

```bash
#!/bin/bash
# Autoresearch loop — runs Junie in a cycle
# Usage: ./autoresearch/run.sh [max_experiments]

MAX=${1:-50}
RESULTS="autoresearch/results.tsv"

# Initialize results file if needed
if [ ! -f "$RESULTS" ]; then
    echo "hash\teval_score\teval_breadth\teval_best_ms\tdescription\tstatus" > "$RESULTS"
fi

# Baseline run
echo "Running baseline eval..."
python autoresearch/eval.py > run.log 2>&1
BASELINE=$(grep "^eval_score:" run.log | awk '{print $2}')
echo "Baseline eval_score: $BASELINE"

# Main loop — Junie handles the inner experiment cycle
echo "Starting Junie autoresearch loop (max $MAX experiments)..."
junie "$(cat autoresearch/program.md)

Current baseline eval_score: $BASELINE
Run up to $MAX experiments. Start now."
```

**Acceptance criteria:**
- [ ] `./autoresearch/run.sh 10` launches Junie and produces results.tsv entries
- [ ] Git log shows experiment commits (kept) and no reverted commits remain
- [ ] results.tsv has one row per experiment with all columns populated

---

## Implementation Order (Weekend Schedule)

### Friday Night (Optional): Pre-download Models
- `huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct`
- `huggingface-cli download deepseek-ai/deepseek-coder-1.3b-instruct`
- Saves ~30min Saturday morning

### Saturday Morning: Foundation (Steps 1-2, ~3 hours)
1. Set up venv with new deps, verify both models download and load on CPU
2. Implement `reward.py` with normalized rewards and graduated penalties
3. Test reward function with synthetic inputs

### Saturday Afternoon: GRPO Core (Step 3, ~4 hours)
4. Implement `grpo_train.py` with manual GRPO loop + CoT-forcing prompt
5. Run 1 iteration on Qwen-1.5B to validate the pipeline end-to-end
6. Fix any OOM or speed issues (adjust batch size, group size)
7. Kick off full Qwen-1.5B training run overnight (~1.5-2.5 hours for 30 iterations)

### Saturday Evening: Autoresearch Setup (Steps 6-8, ~2 hours)
8. Ensure Ollama is running (`docker compose up ollama`)
9. Write frozen `eval.py` harness, validate grepable output
10. Write `program.md` and `run.sh`
11. Test one manual Junie cycle (edit → eval → keep/revert)
12. Start Junie loop on branch `autoresearch/experiments` overnight

### Sunday Morning: Second Model + Review (~3 hours)
13. Review Qwen overnight results + Junie's experiments
14. Adjust `program.md` if Junie got stuck or gamed the metric
15. Start DeepSeek training run (~1.5-2.5 hours)
16. Rewrite `visualize_results.py` for multi-model comparison

### Sunday Afternoon: Analysis (~3 hours)
17. Compare Qwen vs DeepSeek vs baseline prompt-guided approach
18. Compare Junie-optimized classifier vs original thresholds
19. End-to-end test: GRPO with Junie-tuned classifier
20. (Stretch) Start Phi-3.5 training if time permits
21. Write up findings

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU training too slow (>30min/iteration) | Delays full run | Reduce group_size from 4→2, reduce max_new_tokens from 512→256, accept fewer iterations (15 instead of 30) |
| Model OOM on 64GB | Blocks training | Use `torch.float16` (CPU supports it, just slower) or reduce model to 0.5B variant. No `bitsandbytes` — it requires CUDA. |
| Junie modifies eval.py or other files | Invalid experiments | Git pre-commit hook that rejects changes outside `RL/classifer.py`. Also: `program.md` explicitly forbids it |
| Junie gets stuck in a loop | Wasted overnight time | Cap at 50 experiments in `run.sh`, log everything to results.tsv for post-mortem |
| Eval harness variance too high | Can't distinguish good/bad thresholds | Fix random seed in eval.py, run each eval 3x and average |
| Phi-3.5 is 3.8B not 1.5B | 2-3x slower, ~15GB RAM | Stretch goal only. Focus weekend on Qwen + DeepSeek |
| Junie git reset conflicts with GRPO output | Corrupted state | Junie works on branch `autoresearch/experiments`, GRPO writes to `results/` (gitignored). No overlap. |
| Ollama not running for Junie eval | Eval harness fails | `run.sh` checks Ollama health before starting; `docker compose up ollama` as prerequisite |
| HuggingFace model download time | Delays Saturday AM | Pre-download all models Friday night: `huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct` |

---

## Verification Steps

1. **GRPO pipeline smoke test**: 1 iteration of Qwen-1.5B completes, produces valid JSONL log entry, LoRA checkpoint saves
2. **Reward normalization**: Synthetic test — group of [50ms, 100ms, inf, inf] produces rewards centered near 0 with std ~1
3. **CoT generation**: After 5 iterations, >80% of outputs contain `<think>` tags
4. **Multi-model parity**: All 3 models produce comparable log format, visualizer handles all
5. **Junie loop**: 5 experiments complete, results.tsv populated, git log clean (no reverted commits in history)
6. **Classifier improvement**: Junie-tuned thresholds produce eval_score > baseline by >5%
7. **End-to-end**: Run GRPO with Junie-tuned classifier → verify it finds more approaches than GRPO with original classifier

---

## LoRA vs Full Fine-Tune: Why LoRA Is Sufficient

For proving that "RL can discover novel algorithmic approaches on small models":

- LoRA updates the **attention projection matrices** (Q, V) — these control *what the model attends to* and *how it reasons about code structure*. This is exactly what needs to change for the model to shift from "emit the first sieve variant I recall" to "consider multiple approaches and pick the best."
- Research shows LoRA at rank 16 achieves **90-95% of full fine-tune performance** on task-specific adaptation (Hu et al. 2021, confirmed by QLoRA paper).
- The paper's key insight — the knowing-doing gap — is about the model choosing the wrong action despite knowing the right one. LoRA on attention layers directly addresses this by modifying the action-selection pathway.
- Full fine-tune on CPU would be **10-50x slower** (updating 1.5B params vs ~4M LoRA params), pushing a weekend project into a multi-week one.

**Bottom line:** If LoRA + GRPO shows measurable improvement over the prompt-guided baseline (more approaches discovered, faster solutions found), that proves the RL hypothesis. Full fine-tune would be a follow-up for marginal gains.

---

## Changelog (Critic Review, 2026-04-02)

1. **Removed `bitsandbytes`** from deps — requires CUDA, does not work on CPU. OOM mitigation updated to use `torch.float16` fallback instead.
2. **Demoted Phi-3.5-mini to stretch goal** — it's 3.8B params (not 1.5B), ~15GB fp32, 2-3x slower. Weekend scope is Qwen + DeepSeek.
3. **Switched from `trl.GRPOTrainer` to manual GRPO loop** — avoids hidden CUDA assumptions, simpler to debug on CPU, ~50 lines of transparent REINFORCE math.
4. **Added Ollama dependency for Workstream 2** — Junie eval harness uses Ollama for fast prompt-guided eval. `run.sh` checks health before starting.
5. **Added git branch isolation** — Junie works on `autoresearch/experiments` branch to prevent `git reset` from conflicting with GRPO output in `results/`.
6. **Added model pre-download step** (Friday night) to avoid Saturday morning delays.
7. **Added `pandas`/`matplotlib`** to requirements (needed by `visualize_results.py`).
8. **Noted `visualize_results.py` is a rewrite**, not a simple update (zero multi-model support currently).
9. **Fixed timeline arithmetic** — 30 iterations at ~3-5min each = ~1.5-2.5 hours per model, not overnight.
