"""
Tamper-evidence utility for the evaluation Judge (eval/prepare.py).

This script implements a "Hard Lock" by comparing the SHA-256 digest of 
eval/prepare.py against a hardcoded EXPECTED_HASH variable. If they do not
match, the experiment is aborted to prevent tampered evaluation.

Usage:
    1. Run `shasum -a 256 eval/prepare.py` in your terminal.
    2. Paste the result into the EXPECTED_HASH variable below.
    3. The 'run_experiment.py' wrapper will now enforce this lock.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# --- THE HARD LOCK ---
# Replace this with the output of: shasum -a 256 eval/prepare.py
EXPECTED_HASH = "570d9e2a89c805992323f21c350a4048fc947aa8df274ff935772bef641a4243" 

PROJECT_ROOT = Path(__file__).resolve().parent
JUDGE_PATH = PROJECT_ROOT / "eval" / "prepare.py"

def compute_sha256(path: Path) -> str:
    """Computes the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_prepare() -> str:
    """
    Checks if the current Judge matches the hardcoded lock.
    Returns the verified digest on success; raises RuntimeError on failure.
    """
    if not JUDGE_PATH.exists():
        raise FileNotFoundError(f"Judge not found at: {JUDGE_PATH}")

    actual = compute_sha256(JUDGE_PATH)
    
    if EXPECTED_HASH == "PASTE_YOUR_HASH_HERE":
        print("\n[SETUP REQUIRED]")
        print(f"Current Judge Hash: {actual}")
        print("Please copy this hash into verify_integrity.py as EXPECTED_HASH.")
        sys.exit(0)

    if actual != EXPECTED_HASH:
        raise RuntimeError(
            "\n[TAMPER DETECTED]\n"
            "The evaluation logic in eval/prepare.py has been modified.\n"
            f"Expected: {EXPECTED_HASH}\n"
            f"Actual:   {actual}"
        )
    
    return actual

if __name__ == "__main__":
    # If run directly, it will perform a check and print the result
    try:
        digest = verify_prepare()
        print(f"Integrity Verified: {digest}")
    except Exception as e:
        print(str(e))
        sys.exit(1)