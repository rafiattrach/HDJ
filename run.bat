@echo off
:: HDJ RAG Evaluation - Double-click to start
:: This launches the PowerShell run script with the right permissions.

echo Starting Health Data Justice - RAG Evaluation...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1"
