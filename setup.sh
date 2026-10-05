#!/bin/bash
# AI Fitness Trainer — Linux/macOS Setup
set -e

echo "════════════════════════════════════════"
echo " AI Fitness Trainer — Linux/macOS Setup"
echo "════════════════════════════════════════"

python3 --version

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "  Activate:   source .venv/bin/activate"
echo "  Desktop:    python main.py"
echo "  API server: uvicorn server.main_api:app"
