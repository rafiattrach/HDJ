#!/bin/bash
# Setup script for Health Data Justice RAG Evaluation
# Requires: Python 3.12+, Ollama

set -e

echo "=============================================="
echo "🏥 Health Data Justice RAG - Setup"
echo "=============================================="

# Find Python 3.12+
PYTHON=""
for py in python3.13 python3.12; do
    if command -v $py &> /dev/null; then
        PYTHON=$py
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.12+ required. Please install via:"
    echo "   brew install python@3.12"
    echo "   or: pyenv install 3.12"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "✓ Using $PYTHON ($PYTHON_VERSION)"

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install from: https://ollama.ai"
    exit 1
fi
echo "✓ Ollama installed"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv .venv
fi
echo "✓ Virtual environment ready"

# Activate venv
source .venv/bin/activate

# Install/upgrade pip and uv
echo ""
echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip uv

# Install project dependencies
uv pip install --quiet -e .

echo "✓ Dependencies installed"

# Pull Ollama model
echo ""
echo "🤖 Pulling Qwen3-Embedding-4B model (this may take a while)..."
ollama pull qwen3-embedding:4b

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo "=============================================="
echo ""
echo "To run the evaluation:"
echo "  source .venv/bin/activate"
echo "  python evaluate.py"
echo ""
