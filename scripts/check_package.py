"""Install our wheel into a fresh target; reuse the environment's verified dependencies."""
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root/'dist').glob('fly_chess-*.whl'), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise RuntimeError('Build the wheel before checking the package')
    wheel = wheels[-1]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in ('Sans.ttf', 'Mono.ttf', 'OFL-Sans.txt', 'OFL-Mono.txt'):
            assert 'fly_chess/ui/assets/'+name in names
        assert not any('/__pycache__/' in name for name in names)
    home = root/'logs'/('package-check-'+uuid4().hex[:8])
    home.mkdir(parents=True)
    target = home/'installed'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(target), str(wheel)], check=True)
    result = subprocess.run([sys.executable, '-I', str(root/'scripts/verify_release.py'),
        '--package-root', str(target), '--output', str(home/'workflow')], cwd=home,
        capture_output=True, text=True, timeout=180)
    (home/'stdout.txt').write_text(result.stdout)
    (home/'stderr.txt').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    report = json.loads((home/'workflow/verification.json').read_text())
    report['wheel'] = str(wheel)
    report['installation_scope'] = 'Fresh target package; preinstalled Python/native dependencies reused, no clean OS claim'
    (home/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'report': str(home/'verification.json'),
        'installed_package': report['package'], 'orphan_children': report['orphan_children']}, indent=2))


if __name__ == '__main__':
    main()
