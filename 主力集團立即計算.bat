@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
title 主力集團 立即計算（測試版）
echo 計算並上傳「主力集團偵測」（測試版，只有管理員看得到）。只讀本機分點明細、不連富邦，每檔約 4 分鐘。
echo.
".venv\Scripts\python.exe" -m pip install -q pyarrow scikit-learn pyyaml
".venv\Scripts\python.exe" tools\syndicate_now.py %*
echo.
echo 結束，可以關閉這個視窗。重新整理網頁即可看到。
pause
