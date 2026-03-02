# Setup script for Health Data Justice RAG Evaluation
# Requires: Python 3.12+, Ollama

$ErrorActionPreference = "Stop"

Write-Host "=============================================="
Write-Host "Health Data Justice RAG - Setup"
Write-Host "=============================================="

# Find Python 3.12+
$Python = $null
foreach ($py in @("python3.13", "python3.12", "python")) {
    $cmd = Get-Command $py -ErrorAction SilentlyContinue
    if ($cmd) {
        $versionOutput = & $cmd.Source --version 2>&1
        if ($versionOutput -match '(\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 12) {
                $Python = $cmd.Source
                Write-Host "Using $py ($major.$minor)"
                break
            }
        }
    }
}

if (-not $Python) {
    Write-Host "Python 3.12+ required. Please install from: https://www.python.org/downloads/"
    exit 1
}

# Check for Ollama
if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found. Please install from: https://ollama.ai"
    exit 1
}
Write-Host "Ollama installed"

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host ""
    Write-Host "Creating virtual environment..."
    & $Python -m venv .venv
}
Write-Host "Virtual environment ready"

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Install/upgrade pip and uv
Write-Host ""
Write-Host "Installing dependencies..."
python -m pip install --quiet --upgrade pip uv

# Install project dependencies
uv pip install --quiet -e .

Write-Host "Dependencies installed"

# Pull Ollama model
Write-Host ""
Write-Host "Pulling Qwen3-Embedding-4B model (this may take a while)..."
ollama pull qwen3-embedding:4b

Write-Host ""
Write-Host "=============================================="
Write-Host "Setup complete!"
Write-Host "=============================================="
Write-Host ""
Write-Host "To run the evaluation:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python evaluate.py"
Write-Host ""
