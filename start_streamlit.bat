@echo off
echo ========================================
echo DataTobiz Brand Monitoring System
echo Streamlit Web Application
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if we're in the right directory
if not exist "streamlit_app.py" (
    echo ERROR: streamlit_app.py not found
    echo Please run this script from the datatobiz-brand-monitoring directory
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/update dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found
    echo Creating .env file from template...
    if exist "env.template" (
        copy env.template .env
        echo Please edit .env file with your API keys
    ) else (
        echo Please create a .env file with your API keys
    )
)

REM Start Streamlit application
echo.
echo ========================================
echo Starting Streamlit application...
echo Application will be available at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run streamlit_app.py

echo.
echo Application stopped.
pause 