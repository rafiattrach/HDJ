#!/bin/bash
# =============================================================
# HDJ RAG Evaluation - One-Click Launcher
# Run this script to update and start the application.
# =============================================================

# --- Configuration ---
APP_URL="http://localhost:8501"
VENV_DIR=".venv"
STREAMLIT_APP="frontend/app.py"

# --- Helpers ---
print_header() {
    echo ""
    echo "=============================================="
    echo "  Health Data Justice - RAG Evaluation"
    echo "=============================================="
    echo ""
}

print_step() {
    echo "  [$1/5] $2"
}

print_success() {
    echo ""
    echo "=============================================="
    echo "  All good! The app is running."
    echo "  Open in your browser: $APP_URL"
    echo "=============================================="
    echo ""
    echo "  Press Ctrl+C in this window to stop the app."
    echo ""
}

print_error() {
    echo ""
    echo "  ** Problem: $1"
    if [ -n "$2" ]; then
        echo "  ** What to do: $2"
    fi
    echo ""
}

# Move to the script's own directory (so it works from anywhere)
cd "$(dirname "$0")" || {
    print_error "Could not find the project folder." "Make sure you run this script from inside the project."
    exit 1
}

print_header

# ---------------------------------------------------------
# Step 1: Pull latest changes
# ---------------------------------------------------------
print_step 1 "Checking for updates..."

if ! command -v git &> /dev/null; then
    print_error "Git is not installed." "Install Git from https://git-scm.com and try again."
    exit 1
fi

# Stash any local data changes so pull doesn't fail
git stash --quiet 2>/dev/null

PULL_OUTPUT=$(git pull 2>&1)
PULL_EXIT=$?

git stash pop --quiet 2>/dev/null

if [ $PULL_EXIT -ne 0 ]; then
    # Pull failed — but don't block the user, just warn
    echo "       Could not check for updates (no internet?)."
    echo "       Continuing with the current version."
else
    if echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
        echo "       Already up to date."
    else
        echo "       Updated to the latest version."
    fi
fi

# ---------------------------------------------------------
# Step 2: Check Python
# ---------------------------------------------------------
print_step 2 "Checking Python..."

PYTHON=""
for py in python3.13 python3.12 python3; do
    if command -v "$py" &> /dev/null; then
        VERSION=$("$py" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 12 ]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    print_error "Python 3.12 or newer is required but was not found." \
                "Install it with: brew install python@3.13   (then restart your terminal)"
    exit 1
fi
echo "       Found $PYTHON ($VERSION)."

# ---------------------------------------------------------
# Step 3: Set up environment & dependencies
# ---------------------------------------------------------
print_step 3 "Preparing environment..."

if [ ! -d "$VENV_DIR" ]; then
    echo "       Creating virtual environment (first run, one moment)..."
    "$PYTHON" -m venv "$VENV_DIR" 2>&1 || {
        print_error "Could not create the virtual environment." \
                    "Try deleting the .venv folder and running this script again."
        exit 1
    }
fi

# Activate
source "$VENV_DIR/bin/activate" 2>/dev/null || {
    print_error "Could not activate the virtual environment." \
                "Try deleting the .venv folder and running this script again."
    exit 1
}

# Install/update dependencies (quiet unless something goes wrong)
pip install --quiet --upgrade pip uv 2>/dev/null
DEP_OUTPUT=$(uv pip install --quiet -e . 2>&1)
DEP_EXIT=$?

if [ $DEP_EXIT -ne 0 ]; then
    print_error "Could not install dependencies." \
                "Check your internet connection, then try again. If the problem persists, delete the .venv folder and re-run."
    echo "       Details: $DEP_OUTPUT"
    exit 1
fi
echo "       Environment ready."

# ---------------------------------------------------------
# Step 4: Check Ollama
# ---------------------------------------------------------
print_step 4 "Checking Ollama..."

if ! command -v ollama &> /dev/null; then
    print_error "Ollama is not installed." \
                "Download it from https://ollama.com and install it, then run this script again."
    exit 1
fi

# Check if Ollama is running; if not, start it in the background
if ! curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "       Starting Ollama in the background..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!

    # Wait up to 15 seconds for Ollama to come online
    for i in $(seq 1 15); do
        if curl -s --max-time 1 http://localhost:11434/api/tags > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_error "Ollama did not start in time." \
                    "Open Ollama manually (look for the Ollama app), then run this script again."
        exit 1
    fi
fi

# Make sure the embedding model is available
if ! ollama list 2>/dev/null | grep -q "qwen3-embedding:4b"; then
    echo "       Downloading the embedding model (first run, this takes a few minutes)..."
    ollama pull qwen3-embedding:4b 2>&1 || {
        print_error "Could not download the embedding model." \
                    "Check your internet connection and try again."
        exit 1
    }
fi
echo "       Ollama is running."

# ---------------------------------------------------------
# Step 5: Launch the app
# ---------------------------------------------------------
print_step 5 "Starting the application..."

# Open the browser after a short delay (gives Streamlit time to boot)
(sleep 2 && open "$APP_URL" 2>/dev/null) &

print_success

# Run Streamlit (this blocks until the user presses Ctrl+C)
streamlit run "$STREAMLIT_APP" --server.port 8501 2>&1 | while IFS= read -r line; do
    # Filter out noisy Streamlit output, only show important lines
    if echo "$line" | grep -qiE "error|exception|traceback|failed"; then
        echo "  [app] $line"
    fi
done

echo ""
echo "  App stopped. Close this window or run the script again to restart."
echo ""
