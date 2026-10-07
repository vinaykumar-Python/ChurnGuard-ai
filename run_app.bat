@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ChurnGuard] Virtual environment not found.
  echo Run: py -3.13 -m venv .venv
  echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo [ChurnGuard] Starting Flask server...
.venv\Scripts\python.exe app.py
