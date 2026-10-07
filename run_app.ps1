Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Virtual environment not found. Create it with: py -3.13 -m venv .venv" -ForegroundColor Yellow
    exit 1
}
& ".venv\Scripts\python.exe" app.py
