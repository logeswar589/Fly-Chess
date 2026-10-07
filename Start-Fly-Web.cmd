@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Fly.cmd first, then open this launcher again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m fly_chess --config configs\lightweight.toml web %*
set "flyResult=%errorlevel%"
popd
pause
exit /b %flyResult%
