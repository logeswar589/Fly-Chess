# Windows setup

Download the complete repository ZIP from GitHub, extract it, and double-click
`Setup-Fly.cmd`. Do not run it inside the ZIP preview. Wait for **Setup complete**,
then double-click `Start-Fly-Web.cmd` and open the printed access link in Chrome.
Keep that terminal open while using the browser app. `Start-Fly-Desktop.cmd`
launches the desktop interface instead.

The installer supports Windows 10/11 on Intel/AMD 64-bit PCs. It needs internet
and several GB of free space, especially for NVIDIA packages. It reuses a suitable
64-bit Python 3.11–3.14 or installs Python 3.13.13 for the current Windows account.
It creates `.venv`, installs PyTorch and every dependency declared by the project,
then checks dependency consistency, legal chess moves, actual neural inference,
brain measurements, fonts and browser assets. No Node.js or Git is needed.

Python and the Microsoft C++ runtime, when missing, are downloaded from their
official publishers and checked with Windows Authenticode before execution.
The runtime installer may request Windows administrator approval. Python installs
per user without changing PATH. The CMD launcher sets PowerShell's execution
policy for its own process only; it does not change your machine's policy.

## CPU, NVIDIA and optional checks

From a terminal in the extracted project folder:

```bat
Setup-Fly.cmd -Device Cpu
Setup-Fly.cmd -Device Cuda
Setup-Fly.cmd -Dev
Setup-Fly.cmd -VerifyOnly
```

Default `Auto` keeps an existing working PyTorch 2.11.0 installation. On a fresh
installation it chooses CUDA 12.8 when `nvidia-smi` detects an NVIDIA GPU,
otherwise CPU. If the automatic NVIDIA package download/install fails it tries
CPU. Actual device usability is checked during verification; auto mode can fall
back to CPU when the driver or GPU cannot execute the model. The explicit `Cuda`
option requires usable CUDA and reports an error otherwise. Fly does not install
drivers or a system CUDA toolkit. CPU works without an NVIDIA GPU.

`Dev` includes development/test dependencies. `VerifyOnly` checks the existing
environment without installing or upgrading dependencies. Advanced automation
can invoke `powershell.exe -NoProfile -ExecutionPolicy Bypass -File Setup-Fly.ps1`
with these same switches, avoiding the CMD launcher's final pause. `-Python`
accepts a specific Python executable; `-EnvironmentPath` creates/checks an alternate
environment (the normal launchers always use the project's `.venv`).

## Rerunning and troubleshooting

- Close Fly and any Python process using its environment before installing again.
  Setup refuses to update an environment it detects in use.
- Setup preserves `models`, `data` and configuration files. It never silently
  deletes or recreates an invalid `.venv`; rename that folder for backup and rerun.
- Read `logs/setup-*.log` if setup fails. Correct the reported issue and rerun.
  Interrupted downloads or package installs can normally be retried.
- If Windows requests a restart for a runtime installation, restart and rerun.
- If a network proxy blocks downloads, allow the official Python/PyTorch/PyPI
  endpoints or install dependencies manually using the README instructions.
- Managed Windows policies may prohibit scripts or software installation even
  with the launcher's process setting; an administrator must resolve that policy.
- A GitHub download includes source, not your personal trained weights. Training
  is optional for trying the app, and an untrained model will play weak chess.

Browser access defaults to this PC. Other devices require the host PC to remain
running and reachable; see [browser access](browser.md) for LAN use.

Installer sources: [Python 3.13.13](https://www.python.org/downloads/release/python-31313/),
[PyTorch builds](https://pytorch.org/get-started/previous-versions/),
[Microsoft C++ runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/).

Verification performed: a fresh isolated CPU environment installed and passed the
smoke checks on Windows; the existing CUDA environment also passed verification.
The missing-Python and missing-runtime bootstrap paths have not been exercised on
a clean Windows installation.
