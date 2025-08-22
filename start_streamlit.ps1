# DataTobiz Brand Monitoring System - PowerShell Startup Script
# Run this script with: PowerShell -ExecutionPolicy Bypass -File start_streamlit.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DataTobiz Brand Monitoring System" -ForegroundColor Cyan
Write-Host "Streamlit Web Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "streamlit_app.py")) {
    Write-Host "ERROR: streamlit_app.py not found" -ForegroundColor Red
    Write-Host "Please run this script from the datatobiz-brand-monitoring directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install/update dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create logs directory if it doesn't exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found" -ForegroundColor Yellow
    if (Test-Path "env.template") {
        Write-Host "Creating .env file from template..." -ForegroundColor Yellow
        Copy-Item "env.template" ".env"
        Write-Host "Please edit .env file with your API keys" -ForegroundColor Yellow
    } else {
        Write-Host "Please create a .env file with your API keys" -ForegroundColor Yellow
    }
}

# Start Streamlit application
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Streamlit application..." -ForegroundColor Cyan
Write-Host "Application will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    streamlit run streamlit_app.py
} catch {
    Write-Host "ERROR: Failed to start Streamlit application" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit" 