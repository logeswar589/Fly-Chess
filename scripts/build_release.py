"""Reproducible contents, explicit allowlist: never ship personal data or models."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
    output = root/'dist'
    output.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                    '--wheel-dir', str(output), str(root)], check=True)
    source = output/f'fly-chess-{version}-source.zip'
    files = [root/name for name in ('pyproject.toml', 'README.md', 'PHASE_PLAN.md', 'STATUS.md',
                                    'Start-Fly.ps1', 'constraints-tested.txt', '.gitignore')]
    for name in ('src/fly_chess', 'configs', 'docs', 'tests', 'scripts'):
        files.extend(p for p in (root/name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, f'fly-chess-{version}/'+path.relative_to(root).as_posix())
    wheel = output/f'fly_chess-{version}-py3-none-any.whl'
    manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (wheel, source)}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
