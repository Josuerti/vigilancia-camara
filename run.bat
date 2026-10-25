@echo off
REM Lanza la vigilancia usando el entorno virtual del proyecto.
cd /d "%~dp0"
".venv\Scripts\python.exe" detector.py %*
pause
