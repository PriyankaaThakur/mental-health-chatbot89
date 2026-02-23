@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements.txt -q
echo.
echo Starting Mental Health Chatbot...
echo.
python app.py
pause
