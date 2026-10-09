@echo off
title ElevatorGuard - Elevator E-Bike Warning System (port 8300)
rem Prefer conda "yolo" env if it exists, otherwise use python from PATH
if exist "D:\anaconda3\envs\yolo\python.exe" (
    start "ElevatorGuard" "D:\anaconda3\envs\yolo\python.exe" "%~dp0server.py"
) else (
    start "ElevatorGuard" python "%~dp0server.py"
)
echo Starting system... browser will open http://127.0.0.1:8300
echo (If it does not open automatically, visit http://127.0.0.1:8300 manually)
timeout /t 3 >nul
