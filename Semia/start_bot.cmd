@echo off
rem Start the Semia Telegram bridge in the background.
start "" /min "C:\Program Files\Python310\pythonw.exe" "C:\Users\diredix\Downloads\books_qdrant\Semia\bot.py" > "C:\Users\diredix\Downloads\books_qdrant\Semia\bot.log" 2>&1
echo Semia bot started. Log: C:\Users\diredix\Downloads\books_qdrant\Semia\bot.log