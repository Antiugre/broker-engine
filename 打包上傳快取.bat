@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 本地建檔完成後執行：把 5 年歷史資料打包成 data_cache.tar.gz，交給雲端每日增量更新
where gh >nul 2>nul
if %errorlevel%==0 (
  ".venv\Scripts\python.exe" tools\cache_bundle.py upload --repo Antiugre/broker-engine
) else (
  ".venv\Scripts\python.exe" tools\cache_bundle.py pack --out data_cache.tar.gz
  echo.
  echo 已產生 data_cache.tar.gz。請到 GitHub 的 broker-engine 專案 Releases 的「cache」上傳這個檔案。
)
pause
