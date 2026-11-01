# 註冊 Windows 工作排程：每週一～五執行 pipeline.py（下載 → 清洗 → 運算 → 上傳 Firebase）
# 用法：powershell -ExecutionPolicy Bypass -File tools\register_task.ps1 -Dir <專案資料夾> [-Time 18:30] [-Primary | -Backup] [-Remove]
#   -Primary：MandM 主跑 → 每週一～五 18:30 執行 pipeline.py --wait-ready --max-hours 20（雲端 23:00 只做備援）
#   雲端（GitHub Actions）主跑 17:40 起；兩台電腦都用 -Backup → 20:30 執行 pipeline.py --failover
#   （雲端或另一台今天已上傳就自動結束；電腦沒開機則開機後補跑）
param(
  [Parameter(Mandatory = $true)][string]$Dir,
  [string]$Time = "",
  [switch]$Backup,
  [switch]$Primary,
  [switch]$Remove
)
$ErrorActionPreference = "Stop"
$TaskName = if ($Primary) { "券商分點_MandM主跑" } elseif ($Backup) { "券商分點_備援運算" } else { "券商分點_上傳Firebase" }
if (-not $Time) { $Time = if ($Primary) { "18:30" } elseif ($Backup) { "20:30" } else { "17:30" } }
$ArgsLine = if ($Primary) { '"pipeline.py" --wait-ready --max-hours 20' } elseif ($Backup) { '"pipeline.py" --failover --wait-ready' } else { '"pipeline.py" --wait-ready' }
$Limit = if ($Primary) { 22 } else { 6 }
$Dir = (Resolve-Path $Dir).Path

if ($Remove) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "已移除排程：$TaskName"
  exit 0
}

$py = Join-Path $Dir ".venv\Scripts\pythonw.exe"   # pythonw：背景執行，不跳出黑色視窗
if (-not (Test-Path $py)) { throw "找不到 $py，請先執行一次「券商分點.bat」完成安裝" }

$action   = New-ScheduledTaskAction -Execute $py -Argument $ArgsLine -WorkingDirectory $Dir
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
$settings = New-ScheduledTaskSettingsSet `
  -StartWhenAvailable `
  -ExecutionTimeLimit (New-TimeSpan -Hours $Limit) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew
# -StartWhenAvailable：17:30 電腦關機或睡眠，開機後會自動補跑
# --wait-ready：富邦資料還沒更新時每 20 分鐘重試，最多 4 次（約到 18:30）；國定假日會自動放棄、不上傳

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
  -Description "台股券商分點：下載、清洗並上傳 Firebase（$Dir）" -Force | Out-Null

$t = Get-ScheduledTask -TaskName $TaskName
$info = $t | Get-ScheduledTaskInfo
Write-Host "已建立排程：$TaskName"
Write-Host "  時間：每週一～五 $Time"
Write-Host "  下次執行：$($info.NextRunTime)"
Write-Host "  日誌：$Dir\logs\pipeline-YYYYMMDD.log"
