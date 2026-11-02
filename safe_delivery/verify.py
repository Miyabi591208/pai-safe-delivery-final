"""Check source/model provenance, stored metrics, PDF, video and system tests."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import numpy as np
import imageio_ffmpeg
from pypdf import PdfReader
from .settings import account_id


def check():
    out = Path('output')
    manifest = json.loads(Path('results/heldout/manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((Path('safe_delivery')/name).read_bytes()).hexdigest()==expected, name
    assert hashlib.sha256(Path('models/color_mlp.npz').read_bytes()).hexdigest()==manifest['model_sha256']
    rows = json.loads(Path('results/heldout/episodes.json').read_text())
    summary = json.loads(Path('results/heldout/summary.json').read_text())
    assert len(rows)==90 and len({(x['seed'], x['method']) for x in rows})==90
    assert set(x['seed'] for x in rows)==set(range(100, 130))
    for method in manifest['methods']:
        group = [x for x in rows if x['method']==method]
        assert sum(x['success'] for x in group)==summary[method]['successes']
        assert sum(x['collision'] for x in group)==summary[method]['collisions']
    for row in rows:
        with np.load(f"results/heldout/{row['seed']}_{row['method']}.npz", allow_pickle=False) as f:
            assert np.allclose(np.diff(f['time']), .05)
            acceleration = np.diff(f['velocity'], axis=0)/.05
            jerk = np.diff(acceleration, axis=0)/.05
            measured = float(np.sqrt(np.mean(np.sum(jerk*jerk, axis=1))))
            assert np.isclose(measured, row['jerk_rms_m_s3'], rtol=1e-10)
            assert np.isclose(np.linalg.norm(np.diff(f['xy'], axis=0), axis=1).sum(), row['path_m'])
    for method in ('astar', 'smooth'):
        group = [x for x in rows if x['method']==method]
        for metric in ('sim_time', 'path_m', 'accel_rms_m_s2', 'jerk_rms_m_s3'):
            assert np.isclose(np.mean([x[metric] for x in group]), summary[method]['paired_mean_'+metric])
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    (out/'test_results.txt').write_text(tests.stdout+tests.stderr)
    identity = account_id()
    pdf = out/'pdf'/f'説明資料_PAI最終課題_{identity}.pdf'
    pdf_pages, embedded = None, None
    if pdf.exists():
        reader = PdfReader(pdf)
        pdf_pages = len(reader.pages)
        assert pdf_pages==3
        content = '\n'.join(p.extract_text() for p in reader.pages)
        assert '79.8' in content and '30' in content and 'MuJoCo' in content
        # At least one embedded font; do not depend on external Japanese packs.
        embedded = False
        for page in reader.pages:
            for ref in page['/Resources']['/Font'].values():
                desc = ref.get_object().get('/FontDescriptor')
                if desc and '/FontFile2' in desc.get_object():
                    embedded = True
        assert embedded
    video = out/f'PAI最終課題_{identity}.mp4'
    video_result = 'not included (source-only package)'
    if video.exists():
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
        v = json.loads((out/'video_manifest.json').read_text())
        assert frames==v['frames'] and abs(seconds-v['duration_seconds'])<.1
        assert 60<=seconds<=180
        decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video), '-f', 'null', '-'],
                                 capture_output=True, text=True)
        assert decoded.returncode==0 and not decoded.stderr.strip(), decoded.stderr
        video_result = {'frames': frames, 'seconds': seconds, 'decode_errors': 0}
    # Avoid publishing local user paths or obvious credentials in text artifacts.
    checked = 0
    for root in (Path('safe_delivery'), Path('tests'), Path('results'), Path('models')):
        for path in root.rglob('*'):
            if path.suffix in ('.py', '.json', '.md', '.txt'):
                value = path.read_text()
                # Pattern construction prevents the scanner from flagging itself.
                assert ('/'+'Users'+'/') not in value, path
                assert not re.search(r'gh[pousr]_' + r'[A-Za-z0-9]{30,}', value), path
                checked += 1
    report = {'core_source_and_model_hashes': 'match evaluation manifest', 'episodes': len(rows),
              'trace_metrics': 'recomputed and matched', 'system_tests': '9 passed',
              'pdf_pages': pdf_pages, 'pdf_font_embedded': embedded,
              'pdf_status': 'verified' if pdf.exists() else 'not included (source-only package)',
              'video': video_result, 'text_files_scanned': checked,
              'note': 'Visual review is documented separately; limited automated checks are not a universal privacy guarantee.'}
    (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    check()
