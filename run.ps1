# =============================================================
# HDJ RAG Evaluation - One-Click Launcher (Windows)
# Run this script to update and start the application.
# =============================================================

$ErrorActionPreference = "Continue"

# --- Configuration ---
$AppUrl        = "http://localhost:8501"
$VenvDir       = ".venv"
$StreamlitApp  = "frontend/app.py"

# --- Helpers ---
function Print-Header {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "  Health Data Justice - RAG Evaluation"
    Write-Host "=============================================="
    Write-Host ""
}

function Print-Step($num, $msg) {
    Write-Host "  [$num/5] $msg"
}

function Print-Success {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "  All good! The app is running."
    Write-Host "  Open in your browser: $AppUrl"
    Write-Host "=============================================="
    Write-Host ""
    Write-Host "  Press Ctrl+C in this window to stop the app."
    Write-Host ""
}

function Print-Error($problem, $whatToDo) {
    Write-Host ""
    Write-Host "  ** Problem: $problem"
    if ($whatToDo) {
        Write-Host "  ** What to do: $whatToDo"
    }
    Write-Host ""
}

# Move to the script's own directory (so it works from anywhere)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Print-Header

# ---------------------------------------------------------
# Step 1: Pull latest changes
# ---------------------------------------------------------
Print-Step 1 "Checking for updates..."

if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    Print-Error "Git is not installed." "Install Git from https://git-scm.com and try again."
    Read-Host "Press Enter to close"
    exit 1
}

# Stash any local data changes so pull doesn't fail
git stash --quiet 2>$null

$pullOutput = git pull 2>&1 | Out-String
$pullExit = $LASTEXITCODE

git stash pop --quiet 2>$null

if ($pullExit -ne 0) {
    Write-Host "       Could not check for updates (no internet?)."
    Write-Host "       Continuing with the current version."
} elseif ($pullOutput -match "Already up to date") {
    Write-Host "       Already up to date."
} else {
    Write-Host "       Updated to the latest version."
}

# ---------------------------------------------------------
# Step 2: Check Python
# ---------------------------------------------------------
Print-Step 2 "Checking Python..."

$Python = $null
$PythonVersion = ""
foreach ($py in @("python3.13", "python3.12", "python3", "python")) {
    $cmd = Get-Command $py -ErrorAction SilentlyContinue
    if ($cmd) {
        $versionOutput = & $cmd.Source --version 2>&1 | Out-String
        if ($versionOutput -match '(\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 12) {
                $Python = $cmd.Source
                $PythonVersion = "$major.$minor"
                break
            }
        }
    }
}

if (-not $Python) {
    Print-Error "Python 3.12 or newer is required but was not found." `
                "Download it from https://www.python.org/downloads/ and install it (make sure to check 'Add Python to PATH'), then restart this script."
    Read-Host "Press Enter to close"
    exit 1
}
Write-Host "       Found Python $PythonVersion."

# ---------------------------------------------------------
# Step 3: Set up environment & dependencies
# ---------------------------------------------------------
Print-Step 3 "Preparing environment..."

if (-not (Test-Path $VenvDir)) {
    Write-Host "       Creating virtual environment (first run, one moment)..."
    & $Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Print-Error "Could not create the virtual environment." `
                    "Try deleting the .venv folder and running this script again."
        Read-Host "Press Enter to close"
        exit 1
    }
}

# Activate
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Print-Error "Virtual environment is damaged." `
                "Delete the .venv folder and run this script again."
    Read-Host "Press Enter to close"
    exit 1
}
& $ActivateScript

# Install/update dependencies
python -m pip install --quiet --upgrade pip uv 2>$null
$depOutput = uv pip install --quiet -e . 2>&1 | Out-String

if ($LASTEXITCODE -ne 0) {
    Print-Error "Could not install dependencies." `
                "Check your internet connection, then try again. If the problem persists, delete the .venv folder and re-run."
    Write-Host "       Details: $depOutput"
    Read-Host "Press Enter to close"
    exit 1
}
Write-Host "       Environment ready."

# ---------------------------------------------------------
# Step 4: Check Ollama
# ---------------------------------------------------------
Print-Step 4 "Checking Ollama..."

if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Print-Error "Ollama is not installed." `
                "Download it from https://ollama.com and install it, then run this script again."
    Read-Host "Press Enter to close"
    exit 1
}

# Check if Ollama is running; if not, start it in the background
$ollamaRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction SilentlyContinue
    $ollamaRunning = $true
} catch {}

if (-not $ollamaRunning) {
    Write-Host "       Starting Ollama in the background..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden

    # Wait up to 15 seconds for Ollama to come online
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 1 -ErrorAction SilentlyContinue
            $ollamaRunning = $true
            break
        } catch {}
    }

    if (-not $ollamaRunning) {
        Print-Error "Ollama did not start in time." `
                    "Open the Ollama app manually, then run this script again."
        Read-Host "Press Enter to close"
        exit 1
    }
}

# Make sure the embedding model is available
$modelList = ollama list 2>&1 | Out-String
if ($modelList -notmatch "qwen3-embedding:4b") {
    Write-Host "       Downloading the embedding model (first run, this takes a few minutes)..."
    ollama pull qwen3-embedding:4b
    if ($LASTEXITCODE -ne 0) {
        Print-Error "Could not download the embedding model." `
                    "Check your internet connection and try again."
        Read-Host "Press Enter to close"
        exit 1
    }
}
Write-Host "       Ollama is running."

# ---------------------------------------------------------
# Step 5: Launch the app
# ---------------------------------------------------------
Print-Step 5 "Starting the application..."

# Open the browser after a short delay
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process $using:AppUrl
} | Out-Null

Print-Success

# Run Streamlit (this blocks until the user presses Ctrl+C)
try {
    streamlit run $StreamlitApp --server.port 8501
} catch {
    # Ctrl+C triggers this
}

Write-Host ""
Write-Host "  App stopped. Close this window or run the script again to restart."
Write-Host ""
Read-Host "Press Enter to close"
