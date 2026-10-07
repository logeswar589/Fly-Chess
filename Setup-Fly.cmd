@echo off
setlocal
pushd "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup-Fly.ps1" %*
set "flySetupResult=%errorlevel%"
echo.
if not "%flySetupResult%"=="0" echo Setup did not complete. Read the error above and the setup log in logs.
pause
popd
exit /b %flySetupResult%
