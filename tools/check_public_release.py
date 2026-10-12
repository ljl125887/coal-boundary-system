"""Check Git-tracked publication files, without reading ignored runtime data."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT = {
    'README.md', 'pyproject.toml', 'config.example.yaml', '.gitignore',
    '.gitattributes', 'CONTRIBUTING.md',
}
ALLOWED_PREFIXES = ('src/', 'tests/', 'schemas/', 'examples/', 'docs/', 'tools/', '.github/')
FORBIDDEN_SUFFIXES = {
    '.las', '.laz', '.geojson', '.xlsx', '.xls', '.hdf5', '.h5', '.keras',
    '.npz', '.npy', '.pyc', '.jsonl', '.log', '.pem', '.key', '.pfx', '.p12',
    '.png', '.jpg', '.jpeg', '.zip', '.7z',
}
PATTERNS = {
    'private network endpoint': r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b',
    'personal Windows path': r'(?i)[a-z]:[\\/]+Users[\\/]+[^\s"\']+',
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'GitHub token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'AWS access key': r'\bAKIA[0-9A-Z]{16}\b',
    'field coordinate': r'\b(?:119\.01\d{3,}|39\.21\d{3,}|43437\d{2}(?:\.\d+)?|4149\d{2}(?:\.\d+)?)\b',
    'email address': r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',
}


def main() -> int:
    command = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT,
                             check=True, capture_output=True)
    names = [item.decode('utf-8') for item in command.stdout.split(b'\0') if item]
    if not names:
        raise RuntimeError('No tracked files. Stage the reviewed publication files first.')
    issues = []
    for name in names:
        path = ROOT / name
        if name not in ALLOWED_ROOT and not name.startswith(ALLOWED_PREFIXES):
            issues.append((name, 'outside publication allowlist'))
        if path.is_symlink() or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append((name, 'binary/data/link file forbidden'))
            continue
        if path.suffix == '.json' and not (name.startswith('schemas/') or '/schemas/' in name):
            issues.append((name, 'only schema JSON may be published'))
        try:
            content = path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            issues.append((name, 'not readable UTF-8 text'))
            continue
        for label, pattern in PATTERNS.items():
            if re.search(pattern, content):
                issues.append((name, label))
    for name, reason in issues:
        print(f'FAIL {name}: {reason}')  # Do not print matched sensitive contents.
    print(f'Public release check: {len(names)} tracked files, {len(issues)} issues')
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
