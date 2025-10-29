@echo off
echo ========================================
echo Starting ErgonomicXAI Application
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Streamlit web interface...
echo The application will open in your default browser
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run apps\streamlit_viewer.py --server.port 8501 --server.headless false
