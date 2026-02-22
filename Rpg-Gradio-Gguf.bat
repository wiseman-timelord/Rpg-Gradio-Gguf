@echo off
setlocal

title Rpg-Gradio-Gguf
mode con cols=80 lines=25
powershell -noprofile -command "& { $w = $Host.UI.RawUI; $b = $w.BufferSize; $b.Height = 6000; $w.BufferSize = $b; }" >nul 2>&1

:: Change to script's directory
cd /d "%~dp0"

:: Jump to main menu (skip function definitions)
goto :menu

:: Header function - optimized for 79 character width
:header
echo ===============================================================================
echo     %~1
echo ===============================================================================
goto :eof

:menu
rem cls
color 0F
call :header "Rpg-Gradio-Gguf - Main Menu"
echo.
echo.
echo.
echo.
echo.
echo.
echo     1. Launch Rpg-Gradio-Gguf
echo.
echo     2. Launch Rpg-Gradio-Gguf (Debug)
echo.
echo     3. Run Install/Refresh
echo.
echo.
echo.
echo.
echo.
echo.
echo ===============================================================================
set "choice="
set /p "choice=Selection; Menu Option = 1-3, Exit Batch = X: "

if /i "%choice%"=="1" goto :launch
if /i "%choice%"=="2" goto :debug
if /i "%choice%"=="3" goto :install
if /i "%choice%"=="X" goto :exit
if /i "%choice%"=="" goto :menu

echo.
echo Invalid selection. Please try again.
timeout /t 2 >nul
goto :menu

:launch
cls
call :header "Rpg-Gradio-Gguf - Starting Application"
echo.

if not exist "venv\Scripts\pythonw.exe" (
    echo ERROR: Virtual environment not found.
    echo Please run option 3 Installation first.
    echo.
    pause
    goto :menu
)

call .\venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    goto :menu
)
echo Python VENV activated.
echo.

echo Launching program (silent mode)...
echo This console will close. The application is running independently.
echo.

:: Launch with pythonw (no console window) and exit batch entirely
start "" /b pythonw.exe .\launcher.py
timeout /t 2 >nul
exit

:debug
cls
call :header "Rpg-Gradio-Gguf - Starting Application (Debug)"
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Please run option 3 Installation first.
    echo.
    pause
    goto :menu
)

call .\venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    goto :menu
)
echo Python VENV activated.
echo.

echo Launching program (debug mode - console stays open)...
python .\launcher.py

if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%
    echo.
    pause
) else (
    echo.
    echo Application closed normally.
    echo.
    timeout /t 2 >nul
)

call .\venv\Scripts\deactivate.bat >nul 2>&1
echo Python VENV deactivated.
goto :menu

:install
cls
call :header "Rpg-Gradio-Gguf - Installation / Repair"
echo.

:: Use system Python to run installer (venv may not exist yet)
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Please install Python 3.12.x and ensure it is added to PATH.
    echo.
    pause
    goto :menu
)

python .\installer.py

if errorlevel 1 (
    echo.
    echo Installation/repair failed.
    echo Please check the messages above.
    echo.
    pause
) else (
    echo.
    echo Installation/repair finished.
    echo.
    timeout /t 3 >nul
)

goto :menu

:exit
cls
call :header "Rpg-Gradio-Gguf - Goodbye"
echo.
echo Exiting in 2 seconds...
timeout /t 2 >nul
exit
