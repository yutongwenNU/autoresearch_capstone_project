"""
UNIFIED Entry Point: Verifies integrity, runs worker, and grades result.
Usage: 
    python run_experiment.py "description" --baseline
    python run_experiment.py "description" --keep
    python run_experiment.py "description" --discard
"""
import subprocess
import sys
from pathlib import Path
from verify_integrity import verify_prepare

PROJECT_ROOT = Path(__file__).resolve().parent
WORKER = PROJECT_ROOT / "research.py"
JUDGE = PROJECT_ROOT / "eval" / "prepare.py"

def main():
    # 1. Verify Judge Integrity (The TA's "Hard Lock")
    print("=== Step 1/3: Verifying Judge integrity ===")
    try:
        digest = verify_prepare()
        print(f"Judge integrity OK (SHA-256: {digest[:12]}...)")
    except Exception as e:
        print(f"CRITICAL: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Capture Metadata (Description and Status)
    args = sys.argv[1:]
    description = "experiment"
    status = "--keep" # Default
    
    if args:
        # Separate the description string from the flags
        description = args[0]
        if "--baseline" in args: status = "--baseline"
        elif "--discard" in args: status = "--discard"

    # 3. Run the Worker (Agent's research logic)
    print(f"\n=== Step 2/3: Running Worker (research.py) ===")
    worker_proc = subprocess.run([sys.executable, str(WORKER)], cwd=PROJECT_ROOT)
    if worker_proc.returncode != 0:
        print("Worker failed.")
        sys.exit(1)

    # 4. Run the Judge (Evaluation and Logging)
    # We pass the metadata to the judge so it can record them in results.tsv
    print(f"\n=== Step 3/3: Running Judge (eval/prepare.py) ===")
    judge_proc = subprocess.run(
        [sys.executable, str(JUDGE), f"\"{description}\"", status], 
        cwd=PROJECT_ROOT
    )
    
    if judge_proc.returncode == 0:
        print("\nExperiment logged successfully. Check logs/performance.png")
    else:
        print("Judge failed to grade the experiment.")

if __name__ == "__main__":
    main()