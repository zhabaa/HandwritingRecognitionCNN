@echo off
chcp 65001 > nul  :: Устанавливаем кодировку UTF-8
cls

echo Select app to launch:
echo 1 - GUI app (gui_app.app)
echo 2 - Web app (web_app.app)
echo 3 - Train model (modelNN.train)
set /p choice="Enter number [1-3]: "

if "%choice%"=="1" (
    .venv\Scripts\python.exe -m gui_app.app

) else if "%choice%"=="2" (
    .venv\Scripts\python.exe -m web_app.app

) else if "%choice%"=="3" (
    .venv\Scripts\python.exe -m modelNN.train

) else (
    echo Invalid choice!
    pause
    exit /b 1
)

pause
