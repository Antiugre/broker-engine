@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 第一次建立約 5 年歷史資料（Ultra 會員自選 + 成交值前 100 名）。每次最多 6 小時，可重複執行接著補。
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
".venv\Scripts\python.exe" backfill.py --max-hours 6
pause
