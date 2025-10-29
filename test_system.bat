@echo off
echo ========================================
echo ErgonomicXAI - System Test
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Running system test...
python optimized_test.py

echo.
echo Test complete! Check the results above.
echo If you see "System Status: Fully Optimized & Functional", everything is working correctly.
echo.
pause
