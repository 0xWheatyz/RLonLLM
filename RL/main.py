import time
import os
import requests
import json
import re
import traceback
import subprocess
import datetime
import numpy as np
from classifer import BranchManager, Attempt

# --- CONFIGURATION ---
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:0.5b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
TARGET_N = 10**6
MAX_ITERATIONS = 30
GENERATION_BATCH_SIZE = 3
RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/results")
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.jsonl")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.json")

class RLEnvironment:
    """
    Interfaces with the Ollama model and provides evaluation mechanisms.
    """
    def __init__(self, model_name: str, ollama_host: str):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.branch_manager = BranchManager(
            exploit_threshold=0.75,
            reward_delta_threshold=0.02,
            window=4
        )

    def generate_code(self, prompt: str) -> str:
        """
        Calls the model (via Ollama) to generate Python code.
        """
        url = f"{self.ollama_host}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            raw_text = result.get("response", "")
            
            # Extract code blocks
            code_match = re.search(r"```python\n(.*?)```", raw_text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            # Fallback if no markdown
            return raw_text.strip()
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return ""

    def evaluate_code(self, code: str) -> float:
        """
        Executes the generated code and returns a performance score (ms).
        Returns a high penalty (float('inf')) if incorrect or fails.
        """
        if not code:
            return float('inf')

        # 1. Correctness check
        test_script = f"""
{code}

def check():
    try:
        # Basic check
        primes = find_primes(100)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if list(primes) != expected:
            import sys
            # Provide some hint about what's wrong
            if 2 not in primes:
                print("FAILED: Missing prime 2", file=sys.stderr)
            elif len(primes) != len(expected):
                print(f"FAILED: Wrong number of primes. Got {{len(primes)}}, expected {{len(expected)}}", file=sys.stderr)
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
        print((end - start) * 1000) # output ms
    else:
        print("FAILED")
"""
        try:
            # We run it in a subprocess to isolate errors
            res = subprocess.run(
                ["python3", "-c", test_script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if res.returncode != 0:
                print(f" (Exited with code {res.returncode}. Error: {res.stderr.strip()})")
                return float('inf')
            
            output = res.stdout.strip()
            if output == "FAILED":
                print(f" (Failed check. Info: {res.stderr.strip()})")
                return float('inf')
            
            try:
                # In case of multiple lines of output, take the last one
                val = output.split('\n')[-1]
                return float(val)
            except ValueError:
                return float('inf')
                
        except subprocess.TimeoutExpired:
            return float('inf')
        except Exception as e:
            print(f"Eval Error: {e}")
            return float('inf')

    def compute_reward(self, report: dict) -> float:
        """
        Calculates the multi-faceted reward.
        Higher is better.
        """
        # score is ms, so smaller is better.
        current_score = report.get("score", float('inf'))
        
        if current_score == float('inf') or np.isinf(current_score) or np.isnan(current_score):
            return -100.0  # Penalty for failure

        # Base reward on performance (using inverse of score)
        performance_reward = 1000.0 / (current_score + 1.0)

        # Bonus for beating global best
        improvement_bonus = 0.0
        global_best = report.get("global_best", float('inf'))
        if current_score < global_best:
             improvement_bonus = 50.0

        # Penalty for exploitation/repetition
        exploitation_score = report.get("exploitation_score", 0.0)
        if np.isnan(exploitation_score) or np.isinf(exploitation_score):
            exploitation_score = 0.0
        # Increased penalty from 10.0 to 25.0 to more strongly discourage "greedy" micro-optimizations
        exploitation_penalty = exploitation_score * 25.0

        reward = performance_reward + improvement_bonus - exploitation_penalty
        if np.isnan(reward):
            return -100.0
        return reward

def train_loop():
    env = RLEnvironment(MODEL_NAME, OLLAMA_HOST)
    print(f"Starting RL Fine-Tuning for {MODEL_NAME}...")
    print(f"Ollama Host: {OLLAMA_HOST}")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize the log file
    with open(LOG_FILE, "w") as f:
        f.write("")

    best_score_overall = float('inf')
    best_code_overall = ""

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1} ---")

        # 1. Get prompt context from Branch Manager (includes history and constraints)
        history_context = env.branch_manager.get_prompt_context()
        
        system_prompt = (
            "You are an expert Python optimizer. Your task is to write a function `find_primes(n)` "
            "that returns a list of all primes up to n (inclusive). The goal is maximum speed. "
            "Consider optimizations like: Sieve of Eratosthenes, skipping even numbers, "
            "segmented sieves for memory efficiency, or bit manipulation with bytearray. "
            "DO NOT use external libraries like numpy or sympy unless specifically asked. "
            "You MUST output the code within a ```python block. "
            "Do not provide explanations outside the block."
        )
        
        prompt = f"{system_prompt}\n\n{history_context}\n\nWrite a highly optimized find_primes(n) function."

        # 2. Generation (Batch)
        for j in range(GENERATION_BATCH_SIZE):
            print(f"  Attempt {j+1}/{GENERATION_BATCH_SIZE}...", end="", flush=True)
            code = env.generate_code(prompt)
            if not code:
                print(" (Generation failed)")
                continue
                
            score = env.evaluate_code(code)
            if score != float('inf'):
                print(f" Score: {score:.2f}ms")
            # If FAILED, it already printed info inside evaluate_code

            # 3. Record and analyze
            report = env.branch_manager.record(code, score)
            report["score"] = score # Ensure score is in report for reward calculation

            # 4. Feedback
            reward = env.compute_reward(report)

            # Track global best
            if score < best_score_overall:
                best_score_overall = score
                best_code_overall = code
            
            # 5. Log results
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "iteration": i + 1,
                "batch_index": j + 1,
                "score": score,
                "reward": reward,
                "approach_label": report.get("approach_label"),
                "exploitation_score": report.get("exploitation_score"),
                "is_exploiting": report.get("exploiting", False),
                "action": report.get("action"),
                "code": code
            }
            try:
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    f.flush()
            except Exception as e:
                print(f"Error logging: {e}")
            
            if report["action"] == "PRUNE_AND_BRANCH":
                print(f"\n[!!!] {report['message']}")
                break # Move to next iteration to get new context

    print("\nTraining Complete.")
    status_str = env.branch_manager.status()
    print(status_str)

    # Save summary
    summary = {
        "model": MODEL_NAME,
        "iterations": MAX_ITERATIONS,
        "best_score": best_score_overall,
        "best_code": best_code_overall,
        "final_status": status_str,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    train_loop()
