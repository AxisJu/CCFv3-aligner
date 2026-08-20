@echo off
setlocal
cd /d "%~dp0"
title APX100 Mouse Brain Alignment Studio - 3D DAPI Population & Deep Learning Edition
echo ===============================================================================
echo   APX100 Mouse Brain Slice Interactive Alignment Studio
echo   3D DAPI Population Template & Deep Learning Nuclei Segmentation
echo ===============================================================================
echo.

:: Detect Python Environment
set "PYTHON_EXE="
if exist "D:\Environments\anaconda3\envs\spotiflow\python.exe" (
    set "PYTHON_EXE=D:\Environments\anaconda3\envs\spotiflow\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python environment not found!
        echo Please ensure Python is on PATH or configure the path to your conda environment.
        pause
        exit /b 1
    )
)

echo Starting APX100 3D DAPI Alignment Studio...
"%PYTHON_EXE%" scripts\run_studio.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program exited with error code %errorlevel%.
    pause
)
