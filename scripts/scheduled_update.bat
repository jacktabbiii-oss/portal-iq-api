@echo off
REM Portal IQ Scheduled Data Update
REM Schedule this in Windows Task Scheduler for daily execution

cd /d %~dp0..
call .venv\Scripts\activate.bat 2>nul || (
    echo Virtual environment not found, using system Python
)

python scripts/run_updates.py

pause
