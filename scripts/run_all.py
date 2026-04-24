"""
Master end-to-end runner.

Runs all project scripts in dependency order:
    1. run_p3.py   — LSTM training + vol forecasting (generates LSTM preds)
    2. run_p1.py   — RL hedging + baselines (uses LSTM preds from step 1)
    3. run_integration.py — LSTM-guided RL (needs both above)
    4. run_novel.py — novel experiments
    5. build_report_figures.py — assemble all figures for the report

Usage (from project root):
    PYTHONPATH="" /opt/anaconda3/envs/scaf/bin/python scripts/run_all.py

Each script is run in a subprocess so failures are isolated.
"""

import subprocess
import sys
import time
from pathlib import Path

PYTHON  = sys.executable
SCRIPTS = Path(__file__).parent
ORDER   = [
    "run_p3.py",
    "run_p1.py",
    "run_integration.py",
    "run_novel.py",
    "run_backtests.py",       # real-path hedging, TC sensitivity, moneyness stress
    "build_report_figures.py",
]

def run_script(name: str) -> bool:
    script = SCRIPTS / name
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(SCRIPTS.parent),
        env={**__import__("os").environ, "PYTHONPATH": ""},
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n✓  {name} completed in {elapsed:.0f}s")
        return True
    else:
        print(f"\n✗  {name} FAILED (returncode={result.returncode}) after {elapsed:.0f}s")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("END-TO-END PROJECT RUNNER")
    print("=" * 60)
    passed, failed = [], []
    for script in ORDER:
        ok = run_script(script)
        (passed if ok else failed).append(script)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in passed:
        print(f"  PASS  {s}")
    for s in failed:
        print(f"  FAIL  {s}")

    if failed:
        print(f"\n{len(failed)} script(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(passed)} scripts passed.")
