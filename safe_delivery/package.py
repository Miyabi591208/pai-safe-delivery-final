"""Build clean archives without the venv, bytecode, course material or Git metadata."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from .settings import account_id


def build(destination):
    root, out = Path.cwd(), Path(destination).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = account_id()
    files = []
    allowed_roots = {'safe_delivery', 'tests', 'models', 'results', 'output'}
    allowed_suffixes = {'.py', '.json', '.md', '.txt', '.npz', '.png', '.pdf', '.mp4'}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if any(x in ('.venv', '__pycache__', '.git', 'tmp') for x in rel.parts):
            continue
        if len(rel.parts)>1 and rel.parts[0] not in allowed_roots:
            continue
        if p.suffix not in allowed_suffixes and p.name!='.gitignore':
            continue
        if p.suffix in ('.pdf', '.mp4') and identity not in p.name:
            continue  # avoid stale artifacts from a different account ID
        files.append((p, rel))
    packages = []
    for source_only, name in [(False, f'PAI最終課題_{identity}_提出一式.zip'),
                              (True, f'PAI最終課題_{identity}_GitHub用ソース.zip')]:
        target = out/name
        count = 0
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p, rel in files:
                if source_only and (p.suffix in ('.mp4', '.pdf') or rel.as_posix()=='SUBMISSION.md'):
                    continue
                z.write(p, Path('pai-safe-delivery-final')/rel)
                count += 1
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
        packages.append({'file': target.name, 'files': count, 'bytes': target.stat().st_size,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (out/'SHA256SUMS.txt').write_text(''.join(f"{p['sha256']}  {p['file']}\n" for p in packages))
    (out/'package_manifest.json').write_text(json.dumps(packages, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(packages, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='../deliverables')
    build(parser.parse_args().destination)
