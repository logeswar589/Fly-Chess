param(
    [ValidateSet('Auto', 'Cpu', 'Cuda')][string]$Device = 'Auto',
    [string]$EnvironmentPath = '',
    [string]$Python = '',
    [switch]$Dev,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = $PSScriptRoot
$setupExit = 1
$setupTranscript = $false
$setupLock = $null

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}

function Test-Python {
    param([string]$Executable, [string[]]$Prefix = @())
    $probe = 'import sys,struct; assert (3,11) <= sys.version_info[:2] <= (3,14); assert struct.calcsize(''P'')==8; print(sys.executable)'
    try {
        $answer = & $Executable @Prefix -c $probe 2>$null
        if ($LASTEXITCODE -eq 0 -and $answer) {
            $candidate = [string](@($answer)[-1])
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    } catch { }
    return $null
}

function Find-Python {
    if ($Python) {
        $found = Test-Python $Python
        if (-not $found) { throw 'The supplied -Python must be a working 64-bit Python 3.11-3.14 executable.' }
        return $found
    }
    foreach ($version in @('313', '314', '312', '311')) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$version\python.exe"
        if (Test-Path -LiteralPath $candidate) {
            $found = Test-Python $candidate
            if ($found) { return $found }
        }
    }
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        $found = Test-Python 'py.exe' @('-3')
        if ($found) { return $found }
    }
    # Do not launch the Microsoft Store's python.exe installation alias.
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike '*\Microsoft\WindowsApps\*') {
        $found = Test-Python $command.Source
        if ($found) { return $found }
    }
    return $null
}

function Install-Python {
    Write-Host '[1/5] Installing Python 3.13 for your Windows account...'
    # A fixed official installer, verified against Windows' trusted signer chain.
    # No PATH changes, admin requirement, driver installation or global pip changes.
    $version = '3.13.13'
    $downloadDir = Join-Path $projectRoot 'logs\setup-downloads'
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    $installer = Join-Path $downloadDir "python-$version-amd64.exe"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri "https://www.python.org/ftp/python/$version/python-$version-amd64.exe" -OutFile $installer
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*Python Software Foundation*') {
        throw 'Python installer signature could not be verified. Nothing was executed. Install Python 3.13 from python.org, then rerun Setup-Fly.cmd.'
    }
    $target = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313'
    $installArguments = @('/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0', 'Include_test=0', ('TargetDir="' + $target + '"'))
    $process = Start-Process -FilePath $installer -ArgumentList $installArguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 3010)) { throw "Python installation failed with exit $($process.ExitCode)." }
    $found = Test-Python (Join-Path $target 'python.exe')
    if (-not $found) { throw 'Python did not start after installation. Restart Windows if requested, then rerun setup.' }
    return $found
}

function Install-NativeRuntimeIfMissing {
    $systemFolder = [Environment]::GetFolderPath('System')
    if ((Test-Path -LiteralPath (Join-Path $systemFolder 'msvcp140.dll')) -and
        (Test-Path -LiteralPath (Join-Path $systemFolder 'vcruntime140_1.dll'))) { return }
    Write-Host 'Installing the Microsoft C++ runtime required by PyTorch. Windows may request administrator approval.'
    $downloadDir = Join-Path $projectRoot 'logs\setup-downloads'
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    $installer = Join-Path $downloadDir 'vc_redist.x64.exe'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vc14/vc_redist.x64.exe' -OutFile $installer
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*Microsoft Corporation*') {
        throw 'Microsoft runtime signature could not be verified. Nothing was executed.'
    }
    $process = Start-Process -FilePath $installer -ArgumentList @('/install', '/quiet', '/norestart') -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0, 1638, 3010)) { throw "Microsoft runtime installation failed with exit $($process.ExitCode)." }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        throw 'This setup supports Windows 10/11 on Intel/AMD 64-bit PCs. Other platforms require manual setup.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'src\fly_chess\cli.py'))) {
        throw 'Extract the entire GitHub ZIP first. Setup must remain beside pyproject.toml and the src folder.'
    }
    if (-not $EnvironmentPath) { $EnvironmentPath = Join-Path $projectRoot '.venv' }
    $EnvironmentPath = [IO.Path]::GetFullPath($EnvironmentPath)
    $venvPython = Join-Path $EnvironmentPath 'Scripts\python.exe'
    $logDir = Join-Path $projectRoot 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logFile = Join-Path $logDir ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    $setupLock = [IO.File]::Open((Join-Path $logDir 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    Start-Transcript -Path $logFile -Force | Out-Null
    $setupTranscript = $true
    Write-Host 'FLY / CHESS - Windows setup'
    Write-Host "Project: $projectRoot"
    Write-Host "Environment: $EnvironmentPath"
    Write-Host 'Close Fly before installing or updating this environment. Internet is needed for downloads.'
    if (-not $VerifyOnly) {
        $running = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($EnvironmentPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) })
        if ($running.Count) { throw 'Python is currently running from this environment. Close Fly and retry, or use -VerifyOnly to check without changing dependencies.' }
        if (Test-Path -LiteralPath $EnvironmentPath) {
            if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentPath 'pyvenv.cfg')) -or -not (Test-Python $venvPython)) {
                throw 'The environment folder exists but is not a working supported virtual environment. Rename it for backup, then rerun setup; no files were deleted.'
            }
            Write-Host '[1/5] Reusing the existing Python environment.'
        } else {
            $basePython = Find-Python
            if (-not $basePython) { $basePython = Install-Python }
            Write-Host "[2/5] Creating an isolated environment with $basePython"
            Invoke-Checked $basePython @('-m', 'venv', $EnvironmentPath)
        }
        Install-NativeRuntimeIfMissing
        Write-Host '[3/5] Preparing pip and PyTorch (NVIDIA downloads can be several GB)...'
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
        $workingTorch = $false
        if ($Device -eq 'Auto') {
            try {
                & $venvPython -c 'import torch; assert torch.__version__.split(''+'')[0]==''2.11.0''; assert (torch.ones(2,2) @ torch.ones(2,2)).sum().item()==8' 2>$null
                $workingTorch = $LASTEXITCODE -eq 0
            } catch { }
        }
        if (-not $workingTorch) {
            $flavor = 'cpu'
            if ($Device -eq 'Cuda') { $flavor = 'cu128' }
            elseif ($Device -eq 'Auto' -and (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
                try {
                    $gpuOutput = & nvidia-smi.exe -L 2>$null
                    if ($LASTEXITCODE -eq 0 -and $gpuOutput) { $flavor = 'cu128' }
                } catch { }
            }
            try {
                Invoke-Checked $venvPython @('-m', 'pip', 'install', "torch==2.11.0+$flavor", '--index-url', "https://download.pytorch.org/whl/$flavor")
            } catch {
                if ($flavor -ne 'cu128' -or $Device -ne 'Auto') { throw }
                Write-Warning 'NVIDIA package installation failed. Trying the CPU package.'
                Invoke-Checked $venvPython @('-m', 'pip', 'install', 'torch==2.11.0+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu')
            }
        } else { Write-Host 'Keeping the existing working PyTorch build.' }
        Write-Host '[4/5] Installing Fly and all remaining project dependencies...'
        $installTarget = $projectRoot
        if ($Dev) { $installTarget += '[dev]' }
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '-e', $installTarget)
    }
    Write-Host '[5/5] Verifying dependencies, chess, neural inference and browser assets...'
    if (-not (Test-Python $venvPython)) { throw 'No supported environment found. Run Setup-Fly.cmd first.' }
    Invoke-Checked $venvPython @('-m', 'pip', 'check')
    $verifyArguments = @((Join-Path $projectRoot 'scripts\verify_setup.py'))
    if ($Device -eq 'Cuda') { $verifyArguments += '--require-cuda' }
    Invoke-Checked $venvPython $verifyArguments
    Write-Host ''
    Write-Host 'Setup complete.' -ForegroundColor Green
    if ($EnvironmentPath -eq (Join-Path $projectRoot '.venv')) {
        Write-Host 'Browser: double-click Start-Fly-Web.cmd, then open its printed access link in Chrome.'
        Write-Host 'Desktop: double-click Start-Fly-Desktop.cmd.'
    } else { Write-Host "Alternate test environment ready: $venvPython -m fly_chess web" }
    Write-Host 'Fly begins untrained unless you supply a checkpoint. Training is optional for trying the app.'
    Write-Host "Setup log: $logFile"
    $setupExit = 0
} catch {
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Correct the reported issue and rerun Setup-Fly.cmd. Existing models and data are preserved.'
} finally {
    if ($setupTranscript) { Stop-Transcript | Out-Null }
    if ($setupLock) { $setupLock.Dispose() }
}
exit $setupExit
