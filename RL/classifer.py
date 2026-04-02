"""
Exploitation Pattern Classifier for LLM Code Search
=====================================================
Detects when an LLM is exploiting a local optimum vs genuinely exploring.
Classifies both known patterns (sieve variants, etc.) and unknown ones via clustering.
"""

import ast
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# 1. DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class Attempt:
    code: str
    score: float                        # lower is better (e.g. ms runtime)
    timestamp: float = field(default_factory=time.time)
    fingerprint: Optional[str] = None   # set after classification
    approach_label: Optional[str] = None


@dataclass
class BranchNode:
    attempt: Attempt
    children: list = field(default_factory=list)
    exploitation_score: float = 0.0     # rises as reward delta flattens
    is_pruned: bool = False
    visits: int = 1
    reward_history: list = field(default_factory=list)

    def avg_reward(self):
        return np.mean(self.reward_history) if self.reward_history else self.attempt.score


# ---------------------------------------------------------------------------
# 2. STRUCTURAL FINGERPRINTER
# ---------------------------------------------------------------------------

class StructuralFingerprinter:
    """
    Extracts high-level algorithm features from code using AST analysis.
    Two pieces of code with the same fingerprint are taking the same approach,
    even if the surface-level text is different.
    """

    # Known named patterns — add more as you discover them
    KNOWN_PATTERNS = {
        frozenset(["sieve", "boolean", "range"]): "sieve_basic",
        frozenset(["sieve", "bytearray"]): "sieve_bytearray",
        frozenset(["sieve", "bitarray"]): "sieve_bitarray",
        frozenset(["sieve", "segment"]): "sieve_segmented",
        frozenset(["wheel", "factor"]): "wheel_factorization",
        frozenset(["miller", "rabin"]): "miller_rabin",
        frozenset(["trial", "division"]): "trial_division",
        frozenset(["numpy", "sieve"]): "sieve_numpy",
        frozenset(["sympy"]): "sympy_wrapper",
        frozenset(["multiprocess", "sieve"]): "sieve_parallel",
        frozenset(["cython", "sieve"]): "sieve_cython",
        frozenset(["simd"]): "simd_optimized",
        frozenset(["sieve", "even"]): "sieve_skip_even",
        frozenset(["is_prime", "loop"]): "trial_division_basic",
    }

    def extract_features(self, code: str) -> dict:
        """Pull structural features out of code via AST + keyword scan."""
        features = {
            "keywords": set(),
            "uses_numpy": False,
            "uses_bitwise": False,
            "uses_comprehension": False,
            "uses_generator": False,
            "loop_depth": 0,
            "function_count": 0,
            "has_recursion": False,
            "data_structures": set(),
        }

        if not code:
             return features

        code_lower = code.lower()

        # --- keyword scan (fast path for named patterns) ---
        keyword_map = {
            "sieve": ["sieve", "eratosthenes", "atkin"],
            "segment": ["segment", "segmented"],
            "bitarray": ["bitarray", "bit_array"],
            "bytearray": ["bytearray"],
            "boolean": ["bool", "boolean", "[true]"],
            "wheel": ["wheel"],
            "factor": ["factor"],
            "miller": ["miller"],
            "rabin": ["rabin"],
            "trial": ["trial"],
            "division": ["division", "divisible", "%"],
            "numpy": ["numpy", "np."],
            "multiprocess": ["multiprocess", "multithread", "concurrent", "pool"],
            "cython": ["cython", "ctypes", "cffi"],
            "simd": ["simd", "__m256", "intrinsic"],
            "sympy": ["sympy"],
            "even": ["step=2", "range(3,", "range(2,", "::2"],
            "is_prime": ["def is_prime", "is_prime("],
            "loop": ["for ", "while "],
        }
        for tag, hints in keyword_map.items():
            if any(h in code_lower for h in hints):
                features["keywords"].add(tag)

        # --- AST analysis ---
        try:
            tree = ast.parse(code)
        except SyntaxError:
            features["parse_error"] = True
            return features

        features["uses_numpy"] = "numpy" in features["keywords"]

        for node in ast.walk(tree):
            # bitwise ops
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift)):
                features["uses_bitwise"] = True
            # comprehensions
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
                features["uses_comprehension"] = True
            # generators
            if isinstance(node, ast.GeneratorExp):
                features["uses_generator"] = True
            # function count
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features["function_count"] += 1
            # data structure literals
            if isinstance(node, ast.Dict):
                features["data_structures"].add("dict")
            if isinstance(node, ast.Set):
                features["data_structures"].add("set")
            if isinstance(node, ast.List):
                features["data_structures"].add("list")

        # loop depth (rough proxy for algorithmic complexity class)
        features["loop_depth"] = self._max_loop_depth(tree)

        # recursion detection
        features["has_recursion"] = self._has_recursion(tree)

        return features

    def _max_loop_depth(self, tree) -> int:
        max_depth = [0]
        def visit(node, depth):
            if isinstance(node, (ast.For, ast.While)):
                depth += 1
                max_depth[0] = max(max_depth[0], depth)
            for child in ast.iter_child_nodes(node):
                visit(child, depth)
        visit(tree, 0)
        return max_depth[0]

    def _has_recursion(self, tree) -> bool:
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in func_names:
                    return True
        return False

    def fingerprint(self, code: str) -> str:
        """
        Returns a stable hash of the structural feature set.
        Same algorithm = same fingerprint, even across different variable names.
        """
        features = self.extract_features(code)
        # build a canonical string from features
        canonical = "|".join([
            ",".join(sorted(features["keywords"])),
            str(features["uses_numpy"]),
            str(features["uses_bitwise"]),
            str(features["uses_comprehension"]),
            str(features["loop_depth"]),
            str(features["has_recursion"]),
            ",".join(sorted(features["data_structures"])),
        ])
        return hashlib.md5(canonical.encode()).hexdigest()[:12]

    def classify_approach(self, code: str) -> str:
        """
        Returns a human-readable label for the algorithm used.
        Falls back to an 'unknown_HASH' label for novel approaches.
        """
        features = self.extract_features(code)
        keywords = features["keywords"]

        for pattern_keys, label in self.KNOWN_PATTERNS.items():
            if pattern_keys.issubset(keywords):
                return label

        # Unknown — derive a label from whatever keywords we did find
        if keywords:
            return "unknown_" + "_".join(sorted(keywords))[:40]

        # Nothing recognizable — fall back to structural hash
        fp = self.fingerprint(code)
        return f"unknown_{fp}"


# ---------------------------------------------------------------------------
# 3. FEATURE VECTOR (for clustering unknown approaches)
# ---------------------------------------------------------------------------

def code_to_vector(code: str, fingerprinter: StructuralFingerprinter) -> np.ndarray:
    """
    Converts code into a numeric feature vector for cosine similarity /
    clustering. Used to detect novel-but-similar approaches.
    """
    f = fingerprinter.extract_features(code)

    # One-hot for known keywords
    all_keywords = [
        "sieve", "segment", "bitarray", "bytearray", "boolean", "wheel",
        "factor", "miller", "rabin", "trial", "division", "numpy",
        "multiprocess", "cython", "simd", "sympy"
    ]
    keyword_vec = [1.0 if k in f["keywords"] else 0.0 for k in all_keywords]

    # Scalar features (normalized roughly to [0,1])
    scalar_vec = [
        float(f["uses_numpy"]),
        float(f["uses_bitwise"]),
        float(f["uses_comprehension"]),
        float(f["uses_generator"]),
        min(f["loop_depth"] / 5.0, 1.0),
        min(f["function_count"] / 10.0, 1.0),
        float(f["has_recursion"]),
        float("dict" in f["data_structures"]),
        float("set" in f["data_structures"]),
    ]

    return np.array(keyword_vec + scalar_vec, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# 4. EXPLOITATION DETECTOR
# ---------------------------------------------------------------------------

class ExploitationDetector:
    """
    Monitors a stream of attempts on a single branch and signals when
    the model has shifted from exploration to exploitation.
    """

    def __init__(
        self,
        reward_delta_threshold: float = 0.02,   # <2% improvement = flattening
        window: int = 4,                          # look back N attempts
        similarity_threshold: float = 0.92,       # cosine sim to flag as "same"
        min_attempts: int = 5,                    # increased from 3
    ):
        self.reward_delta_threshold = reward_delta_threshold
        self.window = window
        self.similarity_threshold = similarity_threshold
        self.min_attempts = min_attempts
        self.fingerprinter = StructuralFingerprinter()

    def compute_reward_delta_trend(self, scores: list[float]) -> float:
        """
        Returns the average improvement per step over the window.
        Negative = getting worse, near-zero = exploiting.
        Scores are costs (lower = better), so delta is negative when improving.
        """
        valid_scores = [s for s in scores if s != float('inf')]
        if len(valid_scores) < 2:
            return 0.0  # Not enough data, or all failed
        recent = valid_scores[-self.window:]
        if len(recent) < 2:
            return 0.0
        deltas = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        # negative delta = improvement; we want the magnitude
        improvements = [-d for d in deltas]
        return float(np.mean(improvements))

    def is_structurally_similar(
        self,
        code_a: str,
        code_b: str,
    ) -> bool:
        va = code_to_vector(code_a, self.fingerprinter)
        vb = code_to_vector(code_b, self.fingerprinter)
        return cosine_similarity(va, vb) >= self.similarity_threshold

    def detect(self, attempts: list[Attempt]) -> dict:
        """
        Given a list of attempts on a branch, return an exploitation report.

        Returns:
            {
                "exploiting": bool,
                "reason": str,
                "exploitation_score": float,  # 0.0 = pure exploration, 1.0 = full exploit
                "delta_trend": float,
                "structural_similarity": float,
            }
        """
        result = {
            "exploiting": False,
            "reason": "ok",
            "exploitation_score": 0.0,
            "delta_trend": None,
            "structural_similarity": None,
        }

        if len(attempts) < self.min_attempts:
            return result

        scores = [a.score for a in attempts]
        delta_trend = self.compute_reward_delta_trend(scores)
        result["delta_trend"] = delta_trend

        # --- Signal 1: reward flattening ---
        # If average improvement per step is below threshold
        valid_scores = [s for s in scores if s != float('inf')]
        if not valid_scores:
            delta_signal = 1.0 # Everything failed, we are stuck
        else:
            best_score = min(valid_scores)
            relative_improvement = delta_trend / (best_score + 1e-8)
            delta_signal = 1.0 - min(max(relative_improvement / self.reward_delta_threshold, 0.0), 1.0)

        # --- Signal 2: structural similarity of recent attempts ---
        recent_attempts = attempts[-self.window:]
        recent_codes = [a.code for a in recent_attempts if a.code]
        
        if len(recent_codes) < 2:
            avg_sim = 0.0
        else:
            sims = []
            for i in range(len(recent_codes)):
                for j in range(i+1, len(recent_codes)):
                    sims.append(cosine_similarity(
                        code_to_vector(recent_codes[i], self.fingerprinter),
                        code_to_vector(recent_codes[j], self.fingerprinter),
                    ))
            avg_sim = float(np.mean(sims)) if sims else 0.0
            
        result["structural_similarity"] = avg_sim
        sim_signal = min(avg_sim / self.similarity_threshold, 1.0)

        # Handle NaNs in signals
        if np.isnan(delta_signal): delta_signal = 0.0
        if np.isnan(sim_signal): sim_signal = 0.0

        # --- Combined exploitation score ---
        # 0.6 * delta_signal + 0.4 * sim_signal is the original logic.
        # We want to be more patient if sim_signal is high but we are still improving significantly.
        # "Greedy" micro-optimizations usually have high similarity and low (but positive) improvement.
        
        exploitation_score = 0.6 * delta_signal + 0.4 * sim_signal
        
        # New: Detect "Greedy" behavior (High similarity, low improvement)
        # If they are very similar (sim_signal > 0.8) and improvement is tiny (delta_signal > 0.5)
        # We increase the exploitation score to penalize it.
        if sim_signal > 0.8 and delta_signal > 0.5:
             # This is the "greedy" zone
             exploitation_score = max(exploitation_score, 0.8) # Push into prune territory if persistent
        
        # If too many failures in a row, force exploitation score up
        recent_failures = [1 for a in attempts[-self.window:] if a.score == float('inf')]
        if len(recent_failures) >= self.window:
            exploitation_score = 1.0
            result["reason"] = f"Persistent failure: last {self.window} attempts failed."

        result["exploitation_score"] = exploitation_score

        # --- Decision ---
        if exploitation_score > 0.75:
            result["exploiting"] = True
            if delta_signal > sim_signal:
                result["reason"] = (
                    f"Reward flattening: avg improvement {delta_trend:.4f} "
                    f"vs threshold {self.reward_delta_threshold:.4f}"
                )
            else:
                result["reason"] = (
                    f"Structural convergence: avg pairwise similarity "
                    f"{avg_sim:.3f} >= {self.similarity_threshold}"
                )

        return result


# ---------------------------------------------------------------------------
# 5. PATTERN REGISTRY
# ---------------------------------------------------------------------------

class PatternRegistry:
    """
    Maintains a global record of all approaches tried across all branches.
    Prevents the same approach from being re-explored in a different branch.
    Also discovers and names novel unknown patterns via vector clustering.
    """

    def __init__(self, similarity_threshold: float = 0.88):
        self.similarity_threshold = similarity_threshold
        self.fingerprinter = StructuralFingerprinter()

        # fingerprint_hash -> (label, best_score, vector)
        self.known: dict[str, tuple[str, float, np.ndarray]] = {}

        # Pruned approaches — never re-explore these
        self.pruned_fingerprints: set[str] = set()
        self.pruned_labels: set[str] = set()

    def register(self, attempt: Attempt) -> str:
        """
        Register an attempt. Returns its approach label.
        Assigns a label if this is a novel approach.
        """
        fp = self.fingerprinter.fingerprint(attempt.code)
        attempt.fingerprint = fp

        if fp in self.known:
            label, best, vec = self.known[fp]
            # update best score
            if attempt.score < best:
                self.known[fp] = (label, attempt.score, vec)
            attempt.approach_label = label
            return label

        # Check if it's similar to something we already know
        vec = code_to_vector(attempt.code, self.fingerprinter)
        for known_fp, (label, best, known_vec) in self.known.items():
            sim = cosine_similarity(vec, known_vec)
            if sim >= self.similarity_threshold:
                # Treat as same approach
                attempt.fingerprint = known_fp
                attempt.approach_label = label
                if attempt.score < best:
                    self.known[known_fp] = (label, attempt.score, known_vec)
                return label

        # Genuinely new approach
        label = self.fingerprinter.classify_approach(attempt.code)
        self.known[fp] = (label, attempt.score, vec)
        attempt.approach_label = label
        return label

    def prune(self, fingerprint: str):
        """Mark a fingerprint as exhausted — never revisit."""
        self.pruned_fingerprints.add(fingerprint)
        if fingerprint in self.known:
            label = self.known[fingerprint][0]
            self.pruned_labels.add(label)

    def is_pruned(self, code: str) -> bool:
        fp = self.fingerprinter.fingerprint(code)
        if fp in self.pruned_fingerprints:
            return True
        vec = code_to_vector(code, self.fingerprinter)
        for pruned_fp in self.pruned_fingerprints:
            if pruned_fp in self.known:
                _, _, pruned_vec = self.known[pruned_fp]
                if cosine_similarity(vec, pruned_vec) >= self.similarity_threshold:
                    return True
        return False

    def summary(self) -> str:
        lines = ["=== Pattern Registry ==="]
        for fp, (label, best, _) in self.known.items():
            pruned = "✗ PRUNED" if fp in self.pruned_fingerprints else "✓ active"
            lines.append(f"  [{pruned}] {label:40s} best={best:.2f}ms")
        return "\n".join(lines)

    def build_history_prompt(self) -> str:
        """
        Returns a formatted prompt fragment summarizing all known approaches.
        Inject this into the LLM system prompt each iteration.
        """
        active, pruned = [], []
        for fp, (label, best, _) in self.known.items():
            entry = f"  - {label}: best={best:.2f}ms"
            if fp in self.pruned_fingerprints:
                pruned.append(entry)
            else:
                active.append(entry)

        parts = ["APPROACH HISTORY:"]
        if active:
            parts.append("Active (can be improved):")
            parts.extend(active)
        if pruned:
            parts.append("Closed (do NOT revisit these):")
            parts.extend(pruned)
        if self.pruned_labels:
            parts.append(
                f"\nEXPLICIT CONSTRAINT: Do not use any of these approaches: "
                f"{', '.join(sorted(self.pruned_labels))}"
            )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# 6. BRANCH MANAGER (ties it all together)
# ---------------------------------------------------------------------------

class BranchManager:
    """
    Manages the full tree of search branches.
    Call .record() after each attempt, .should_prune() to check,
    and .get_prompt_context() to build the next LLM prompt.
    """

    def __init__(
        self,
        exploit_threshold: float = 0.75,
        reward_delta_threshold: float = 0.02,
        window: int = 4,
    ):
        self.registry = PatternRegistry()
        self.detector = ExploitationDetector(
            reward_delta_threshold=reward_delta_threshold,
            window=window,
        )
        self.exploit_threshold = exploit_threshold

        # branch_id -> list of attempts
        self.branches: dict[str, list[Attempt]] = defaultdict(list)
        self.current_branch = "root"
        self.branch_counter = 0

        self.global_best: Optional[Attempt] = None
        self.pruned_branches: set[str] = set()

    def record(self, code: str, score: float) -> dict:
        """
        Record a new attempt. Returns status dict with action recommendation.
        """
        attempt = Attempt(code=code, score=score)
        label = self.registry.register(attempt)
        self.branches[self.current_branch].append(attempt)

        if self.global_best is None or score < self.global_best.score:
            self.global_best = attempt

        report = self.detector.detect(self.branches[self.current_branch])
        report["approach_label"] = label
        report["current_branch"] = self.current_branch
        report["global_best"] = self.global_best.score

        if report["exploiting"]:
            # Prune current branch
            self.pruned_branches.add(self.current_branch)
            if attempt.fingerprint:
                self.registry.prune(attempt.fingerprint)

            # Open a new branch
            self.branch_counter += 1
            new_branch = f"branch_{self.branch_counter}"
            self.current_branch = new_branch
            report["action"] = "PRUNE_AND_BRANCH"
            report["new_branch"] = new_branch
            report["message"] = (
                f"Exploitation detected on '{label}': {report['reason']}. "
                f"Opening {new_branch}. Inject closed-path constraints into prompt."
            )
        else:
            report["action"] = "CONTINUE"
            report["message"] = f"Exploring '{label}', exploitation_score={report['exploitation_score']:.2f}"

        return report

    def get_prompt_context(self) -> str:
        """Build the system prompt fragment to inject before each LLM call."""
        return self.registry.build_history_prompt()

    def status(self) -> str:
        lines = [
            f"Current branch : {self.current_branch}",
            f"Total branches : {self.branch_counter + 1}",
            f"Pruned branches: {len(self.pruned_branches)}",
            f"Global best    : {self.global_best.score:.2f}ms ({self.global_best.approach_label})"
            if self.global_best else "Global best    : none",
            "",
            self.registry.summary(),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Simulate a sequence of code attempts the LLM might produce
    ATTEMPTS = [
        # Branch 1: sieve → bytearray micro-optimizations (should trigger pruning)
        ("def sieve(n):\n    is_prime = [True]*(n+1)\n    for i in range(2, int(n**0.5)+1):\n        if is_prime[i]:\n            for j in range(i*i, n+1, i): is_prime[j]=False\n    return [i for i in range(2,n+1) if is_prime[i]]", 120.0),
        ("def sieve(n):\n    is_prime = bytearray([1])*(n+1)\n    for i in range(2, int(n**0.5)+1):\n        if is_prime[i]:\n            for j in range(i*i, n+1, i): is_prime[j]=0\n    return [i for i in range(2,n+1) if is_prime[i]]", 100.0),
        ("def sieve(n):\n    is_prime = bytearray([1])*(n+1)\n    is_prime[0]=is_prime[1]=0\n    for i in range(3, int(n**0.5)+1, 2):\n        if is_prime[i]:\n            is_prime[i*i::2*i]=b'\\x00'*(len(is_prime[i*i::2*i]))\n    return [2]+[i for i in range(3,n+1,2) if is_prime[i]]", 99.0),
        ("def sieve(n):\n    is_prime = bytearray([1])*(n+1)\n    is_prime[0]=is_prime[1]=0\n    for i in range(3, int(n**0.5)+2, 2):\n        if is_prime[i]:\n            is_prime[i*i::2*i]=bytes(len(is_prime[i*i::2*i]))\n    return [2]+[i for i in range(3,n+1,2) if is_prime[i]]", 98.5),

        # Branch 2 (after pruning): numpy approach — genuinely different
        ("import numpy as np\ndef sieve(n):\n    is_prime = np.ones(n+1, dtype=bool)\n    is_prime[:2] = False\n    for i in range(2, int(n**0.5)+1):\n        if is_prime[i]: is_prime[i*i::i] = False\n    return np.nonzero(is_prime)[0].tolist()", 55.0),
        ("import numpy as np\ndef sieve(n):\n    is_prime = np.ones(n+1, dtype=np.uint8)\n    is_prime[:2] = 0\n    for i in range(3, int(n**0.5)+1, 2):\n        if is_prime[i]: is_prime[i*i::2*i] = 0\n    return [2] + np.nonzero(is_prime[3::2])[0].tolist()", 42.0),

        # Branch 2 starts exploiting numpy micro-opts
        ("import numpy as np\ndef sieve(n):\n    sieve_arr = np.ones((n+1)//2, dtype=np.bool_)\n    for i in range(1, int(n**0.5)//2+2):\n        if sieve_arr[i]: sieve_arr[2*i*i+6*i+3::2*i+3] = False\n    return np.r_[2, 2*np.nonzero(sieve_arr)[0][1:]+3].tolist()", 41.8),
        ("import numpy as np\ndef sieve(n):\n    sieve_arr = np.zeros((n+1)//2+1, dtype=np.bool_)\n    sieve_arr[1:] = True\n    for i in range(1, int(n**0.5)//2+2):\n        if sieve_arr[i]: sieve_arr[2*i*(i+1)::2*i+1] = False\n    return [2]+[2*i+1 for i,v in enumerate(sieve_arr) if v and i>0]", 41.5),
    ]

    print("=" * 60)
    print("EXPLOITATION DETECTOR DEMO")
    print("=" * 60)

    manager = BranchManager(
        exploit_threshold=0.90,
        reward_delta_threshold=0.05,
        window=4,
    )

    for i, (code, score) in enumerate(ATTEMPTS):
        print(f"\n[Attempt {i+1}] score={score}ms")
        report = manager.record(code, score)
        print(f"  Approach      : {report['approach_label']}")
        print(f"  Exploit score : {report['exploitation_score']:.2f}")
        print(f"  Delta trend   : {report.get('delta_trend')}")
        print(f"  Similarity    : {report.get('structural_similarity')}")
        print(f"  Action        : {report['action']}")
        if report["action"] == "PRUNE_AND_BRANCH":
            print(f"  *** {report['message']} ***")
            print(f"\n  --- PROMPT CONTEXT FOR NEXT CALL ---")
            print(manager.get_prompt_context())
            print(f"  ------------------------------------")

    print("\n")
    print(manager.status())
