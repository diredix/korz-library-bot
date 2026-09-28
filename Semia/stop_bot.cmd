@echo off
rem Stop the Semia Telegram bridge.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" | Where-Object { $_.CommandLine -like '*books_qdrant*Semia*bot.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo Semia bot stopped (if it was running).