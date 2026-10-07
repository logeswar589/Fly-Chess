@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Fly.cmd first, then open this launcher again.
  pause
  popd
  exit /b 1
)
".venv\Scripts\python.exe" -m fly_chess --config configs\lightweight.toml gui %*
set "flyResult=%errorlevel%"
if not "%flyResult%"=="0" pause
popd
exit /b %flyResult%
