# Research Plan: RL Fine-Tuning of Qwen-Coder-0.8B for Prime Finding

## 1. Hypothesis
Reinforcement learning is an effective way to efficiently train small edge models on one specific coding task. Specifically, we aim to use RL to fine-tune a Qwen-Coder-0.8B model to build a Python application capable of finding primes, optimized for performance and algorithmic diversity.

## 2. Model Selection
- **Model:** Qwen-Coder-0.8B
- **Reasoning:** Small enough for efficient edge deployment and fast RL iterations, while having sufficient pre-training on code to understand Python syntax and basic algorithms.

## 3. RL Strategy: PPO with AST-based Reward Shaping
We will use Proximal Policy Optimization (PPO) or a similar policy-gradient method, but the key innovation is the **Structural Classifier Reward**.

### 3.1 RL Environment
- **Action Space:** Token generation for a Python function `find_primes(n)`.
- **Observation Space:** The problem description and previous (unsuccessful or sub-optimal) attempts in the current branch.
- **State:** The current partial or complete code generated.

### 3.2 Reward Functions
Rewards will be multi-faceted to prevent the model from getting stuck in local optima:
1.  **Correctness (Hard Constraint):** Does the code correctly identify primes for a set of test cases? (Binary: 0 or 1).
2.  **Performance (Efficiency):** Execution time on large `n`. Reward is inversely proportional to execution time.
3.  **Algorithmic Diversity (Classifier-based):**
    -   Using an **AST-based Structural Fingerprinter** to identify the algorithm used (e.g., Trial Division, Sieve of Eratosthenes, Segmented Sieve, Miller-Rabin).
    -   **Negative Reward for Repetition:** If the model generates code with a fingerprint already seen in the current branch or recently pruned branches, it receives a penalty.
    -   **Exploitation Penalty:** Using an **Exploitation Detector** to measure when improvements within a specific algorithmic family (e.g., micro-optimizing a sieve) have plateaued.

## 4. Search Tree and Branching
To prevent the model from "looping" on a single approach, we implement a **Branch Manager**:
-   **Nodes:** Represent distinct algorithmic approaches (fingerprints).
-   **Pruning:** When the Exploitation Detector signals that an approach is exhausted (low reward delta and high structural similarity), that branch is "pruned".
-   **Reverting/Branching:** The system forces the model to backtrack to a higher node in the search tree and provides a prompt context explicitly forbidding the pruned algorithmic fingerprint.

## 5. Autonomous Training Pipeline
1.  **Generation Phase:** Model generates $K$ candidate solutions.
2.  **Evaluation Phase:**
    -   Run code against unit tests for correctness.
    -   Benchmark successful code for performance.
    -   Classify the approach using the `StructuralFingerprinter`.
3.  **Feedback Loop:**
    -   Update the `PatternRegistry` with the new approach and score.
    -   Check for exploitation; if detected, update the `BranchManager` to prune and signal a branch switch.
    -   Update RL policy (PPO) based on the calculated multi-faceted reward.
4.  **Prompt Injection:** For the next generation cycle, inject the "Approach History" into the system prompt to guide the model away from dead ends.

## 6. Evaluation Metrics
-   **Top Execution Speed:** Fastest execution time achieved for $n=10^7$.
-   **Algorithmic Breadth:** Number of distinct, correct algorithmic approaches discovered.
-   **Efficiency of Discovery:** Number of training iterations required to find a superior algorithm compared to the baseline.
-   **Model Size/Latency:** Final performance on edge-equivalent hardware (simulated or actual).
