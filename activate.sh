#!/bin/bash
# Activate script for rl_chess_agent project
# Usage: source activate.sh

# Set up pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Activate virtual environment
source .venv/bin/activate

echo "✓ Virtual environment activated (Python 3.11)"
echo "✓ Project: rl_chess_agent"
