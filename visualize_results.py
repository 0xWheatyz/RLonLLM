"""
Results visualization — supports both single-model and multi-model comparison.

Usage:
    python visualize_results.py                         # single model (results/training_log.jsonl)
    python visualize_results.py --compare               # compare all models in results/
    python visualize_results.py --model qwen-1.5b       # specific model
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = "results"
MODEL_COLORS = {
    "qwen-1.5b": "#2196F3",
    "deepseek-1.3b": "#4CAF50",
    "phi-3.5": "#FF9800",
    "baseline": "#9E9E9E",
}


def load_log(log_path: str) -> pd.DataFrame:
    """Load a training_log.jsonl file into a DataFrame."""
    data = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(data)
    for col in ["score", "reward", "exploitation_score", "loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Replace inf with NaN for plotting
    df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    return df


def load_all_models() -> dict:
    """Discover and load all model result directories."""
    models = {}
    if not os.path.isdir(RESULTS_DIR):
        return models
    for entry in os.listdir(RESULTS_DIR):
        log_path = os.path.join(RESULTS_DIR, entry, "training_log.jsonl")
        if os.path.isfile(log_path):
            try:
                models[entry] = load_log(log_path)
            except Exception as e:
                print(f"Warning: could not load {log_path}: {e}", file=sys.stderr)
    # Also check root results dir for baseline
    root_log = os.path.join(RESULTS_DIR, "training_log.jsonl")
    if os.path.isfile(root_log):
        try:
            models["baseline"] = load_log(root_log)
        except Exception as e:
            print(f"Warning: could not load baseline log: {e}", file=sys.stderr)
    return models


def plot_single(df: pd.DataFrame, output_path: str, title_suffix: str = ""):
    """Original 3-subplot visualization for a single model."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(f"Training Results{' — ' + title_suffix if title_suffix else ''}", fontsize=14)

    valid = df[df["score"].notna()]
    axes[0].plot(valid.index, valid["score"], marker="o", linestyle="-", color="b")
    axes[0].set_title("Score (ms) over Attempts")
    axes[0].set_ylabel("Score (ms, lower=better)")
    axes[0].grid(True)

    axes[1].plot(df.index, df["reward"], marker="s", linestyle="-", color="g")
    axes[1].set_title("Reward over Attempts")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True)

    if "exploitation_score" in df.columns:
        axes[2].plot(df.index, df["exploitation_score"], marker="^", linestyle="-", color="r")
        axes[2].set_title("Exploitation Score over Attempts")
        axes[2].set_ylabel("Exploitation Score")
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


def plot_comparison(models: dict, output_path: str):
    """Side-by-side multi-model comparison plot."""
    if not models:
        print("No model results found to compare.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    fig.suptitle("Multi-Model GRPO Comparison", fontsize=16)

    summary_rows = []

    for model_name, df in sorted(models.items()):
        color = MODEL_COLORS.get(model_name, "#607D8B")
        valid = df[df["score"].notna()]

        # Best score progression (running minimum)
        if not valid.empty:
            running_best = valid["score"].cummin()
            axes[0].plot(
                range(len(running_best)), running_best,
                label=model_name, color=color, linewidth=2,
            )

        # Reward curve
        if "reward" in df.columns:
            axes[1].plot(
                df.index, df["reward"].rolling(5, min_periods=1).mean(),
                label=model_name, color=color, linewidth=2, alpha=0.8,
            )

        # Exploitation score
        if "exploitation_score" in df.columns:
            axes[2].plot(
                df.index, df["exploitation_score"].rolling(5, min_periods=1).mean(),
                label=model_name, color=color, linewidth=2, alpha=0.8,
            )

        # Summary stats
        best_ms = valid["score"].min() if not valid.empty else float("nan")
        approaches = df["approach_label"].nunique() if "approach_label" in df.columns else 0
        # Iteration of best score
        if not valid.empty:
            best_idx = valid["score"].idxmin()
            iter_to_best = df.loc[:best_idx].shape[0]
        else:
            iter_to_best = float("nan")
        summary_rows.append({
            "model": model_name,
            "best_time_ms": round(best_ms, 2),
            "approaches_found": approaches,
            "iterations_to_best": iter_to_best,
            "total_attempts": len(df),
        })

    axes[0].set_title("Best Runtime Over Attempts (running min)")
    axes[0].set_ylabel("Best Score (ms)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Reward (5-attempt rolling mean)")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title("Exploitation Score (5-attempt rolling mean)")
    axes[2].set_ylabel("Exploitation Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

    # Print comparison table
    if summary_rows:
        print("\n=== Model Comparison ===")
        summary_df = pd.DataFrame(summary_rows).set_index("model")
        print(summary_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GRPO training results")
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    parser.add_argument("--model", type=str, default=None, help="Specific model name")
    args = parser.parse_args()

    if args.compare:
        models = load_all_models()
        if not models:
            print(f"No model results found in {RESULTS_DIR}/")
            sys.exit(1)
        plot_comparison(models, os.path.join(RESULTS_DIR, "comparison.png"))
    elif args.model:
        log_path = os.path.join(RESULTS_DIR, args.model, "training_log.jsonl")
        if not os.path.exists(log_path):
            print(f"Log not found: {log_path}")
            sys.exit(1)
        df = load_log(log_path)
        out = os.path.join(RESULTS_DIR, args.model, "visualization.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_single(df, out, title_suffix=args.model)
    else:
        # Legacy single-model mode
        log_path = os.path.join(RESULTS_DIR, "training_log.jsonl")
        if not os.path.exists(log_path):
            print(f"Log file {log_path} not found.")
            sys.exit(1)
        df = load_log(log_path)
        plot_single(df, os.path.join(RESULTS_DIR, "visualization.png"))
