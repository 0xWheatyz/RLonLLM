# Classifier Optimization Program

## Your Task
You are optimizing `RL/classifer.py` to improve algorithmic search effectiveness.
You may ONLY modify numeric thresholds and constants in `RL/classifer.py`.
You may NOT modify `autoresearch/eval.py` or any other file.

## The Metric
After each change, run: `python autoresearch/eval.py`
Extract: `grep "^eval_score:" run.log`
Higher eval_score = better. Current baseline is in autoresearch/results.tsv.

## What You Can Change (in RL/classifer.py)
- `ExploitationDetector.__init__`: `reward_delta_threshold`, `window`, `similarity_threshold`, `min_attempts`
- `BranchManager.__init__`: `exploit_threshold`, `reward_delta_threshold`, `window`
- `PatternRegistry.__init__`: `similarity_threshold`
- `ExploitationDetector.detect`: the `0.6`/`0.4` signal weights, the `0.8`/`0.5` greedy zone thresholds
- `cosine_similarity` threshold in `is_pruned()` (in PatternRegistry)

## Experiment Loop
1. Read current `RL/classifer.py` and `autoresearch/results.tsv`
2. Form a hypothesis about which threshold change will improve eval_score
3. Edit `RL/classifer.py` with ONE threshold change
4. `git add RL/classifer.py && git commit -m "experiment: <description>"`
5. Run: `python autoresearch/eval.py > run.log 2>&1`
6. Extract: `grep "^eval_score:\|^eval_breadth:\|^eval_best_ms:" run.log`
7. If improved: append to results.tsv, keep the commit
8. If equal/worse: `git reset --hard HEAD~1`, log as "reverted" in results.tsv
9. REPEAT — do not stop or ask questions

## Rules
- ONE change per experiment (isolate variables)
- Log every experiment to autoresearch/results.tsv (hash, score, description, status)
- If 3 consecutive experiments show no improvement, try a DIFFERENT threshold
- Never modify autoresearch/eval.py
- Never modify any file outside RL/classifer.py

## results.tsv Format
Tab-separated. Append one row per experiment:
```
git_hash\teval_score\teval_breadth\teval_best_ms\tdescription\tstatus
```
Status is either `kept` or `reverted`.
