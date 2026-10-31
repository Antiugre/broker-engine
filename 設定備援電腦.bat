@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 兩台電腦（MandM、Nunu）都執行一次：每週一～五 20:30 檢查雲端今天是否已完成，沒有才由這台接手
REM 會先移除舊的 17:30 主跑排程（改由雲端主跑）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\register_task.ps1 -Dir . -Remove
powershell -NoProfile -ExecutionPolicy Bypass -File tools\register_task.ps1 -Dir . -Backup
pause
