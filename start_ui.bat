@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0start_ui.ps1" %*
exit /b %errorlevel%
