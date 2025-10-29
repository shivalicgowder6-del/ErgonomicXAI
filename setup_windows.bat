@echo off
echo ========================================
echo ErgonomicXAI - Windows Setup Script
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 from https://python.org
    pause
    exit /b 1
)

echo Python found! Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo 1. Run: start_app.bat
echo 2. Or manually: venv\Scripts\activate && streamlit run apps\streamlit_viewer.py
echo.
echo The web interface will be available at: http://localhost:8501
echo.
pause
