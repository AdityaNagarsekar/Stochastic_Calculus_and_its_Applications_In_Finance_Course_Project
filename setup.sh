#!/bin/bash
# One-time setup: create conda env and install the dependencies used by the
# training scripts, data pipeline, and notebooks.

set -e

ENV_NAME="scaf"

echo "==> Creating conda env '$ENV_NAME' with Python 3.11 ..."
conda create -y -n "$ENV_NAME" python=3.11

echo "==> Activating and installing packages ..."
# Use 'conda run' so the script works non-interactively.
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo ""
echo "===================================================="
echo "Setup complete.  Activate with:  conda activate $ENV_NAME"
echo "Core scripts available:"
echo "  python scripts/run_all.py"
echo "  python scripts/run_p3.py"
echo "  python scripts/run_backtests.py"
echo "Jupyter is also installed:       jupyter notebook"
echo "===================================================="
