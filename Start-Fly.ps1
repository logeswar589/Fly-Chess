param(
    [string]$Workspace = $PSScriptRoot,
    [string]$Weights = '',
    [string]$Config = ''
)
$ErrorActionPreference = 'Stop'
$flyPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $flyPython)) {
    throw 'Set up the environment first: python -m venv .venv, then .\.venv\Scripts\python.exe -m pip install -e ".[dev]"'
}
$flyArguments = @('-m', 'fly_chess', '--workspace', $Workspace)
if ($Config) { $flyArguments += @('--config', $Config) }
$flyArguments += 'gui'
if ($Weights) { $flyArguments += @('--weights', $Weights) }
& $flyPython @flyArguments
exit $LASTEXITCODE
