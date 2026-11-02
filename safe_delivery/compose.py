"""Attach user-supplied course OP/ED with short dissolves; preserve 1x playback."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import imageio.v2 as imageio
import imageio_ffmpeg
from .settings import account_id


def compose(opening, ending):
    out = Path('output')
    target = out / f'PAI最終課題_{account_id()}.mp4'
    manifest_path = out/'video_manifest.json'
    base = json.loads(manifest_path.read_text())
    if 'composition' in base:
        raise ValueError('Regenerate the core video with safe_delivery.media before composing again.')
    inputs = [Path(opening).resolve(), target.resolve(), Path(ending).resolve()]
    meta = []
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(p.name)
        reader = imageio.get_reader(str(p))
        meta.append(reader.get_meta_data())
        reader.close()
    # Current supplied jingles have no audio. Fail explicitly for other assets.
    if any(m.get('audio_codec') for m in meta):
        raise ValueError('This composition is for silent supplied jingles; audio requires a separate mix.')
    fps, dissolve = 30, .4
    durations = [math.ceil(float(m['duration'])*fps-1e-8)/fps for m in meta]
    durations[1] = base['duration_seconds']
    offset_one = durations[0]-dissolve
    offset_two = durations[0]+durations[1]-2*dissolve
    filters = []
    for i, duration in enumerate(durations):
        filters.append(f'[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,'
                       f'pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,'
                       f'fps={fps},tpad=stop_mode=clone:stop_duration=1,'
                       f'trim=duration={duration:.9f},settb=1/{fps},setpts=PTS-STARTPTS,'
                       f'format=yuv420p,fps={fps}[v{i}]')
    filters += [f'[v0][v1]xfade=transition=fade:duration={dissolve}:offset={offset_one:.9f},fps={fps}[first]',
                f'[first][v2]xfade=transition=fade:duration={dissolve}:offset={offset_two:.9f},format=yuv420p[final]']
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix='pai_compose_') as tmp:
        composed = Path(tmp)/'composed.mp4'
        args = [ffmpeg, '-hide_banner', '-v', 'error', '-y']
        for p in inputs:
            args += ['-i', str(p)]
        args += ['-filter_complex', ';'.join(filters), '-map', '[final]', '-an',
                 '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-r', str(fps),
                 '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(composed)]
        subprocess.run(args, check=True)
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(composed))
        expected = sum(durations)-2*dissolve
        assert abs(seconds-expected)<.11, (seconds, expected)
        assert 60<=seconds<=180
        reader = imageio.get_reader(str(composed))
        assert reader.get_meta_data()['size']==(1280, 720)
        reader.close()
        shutil.copyfile(composed, target)
    chapters = [{'title':'配布OP', 'start_seconds':0., 'duration_seconds':durations[0]}]
    for chapter in base['chapters']:
        chapters.append({**chapter, 'start_seconds':round(chapter['start_seconds']+offset_one, 6)})
    chapters.append({'title':'配布ED', 'start_seconds':round(offset_two, 6), 'duration_seconds':durations[2]})
    final = {**base, 'fps':fps, 'frames':frames, 'duration_seconds':seconds,
             'audio':'none; supplied OP/ED also have no audio', 'chapters':chapters,
             'composition':{'core_duration_seconds':base['duration_seconds'], 'core_fps':base['fps'],
                            'dissolve_seconds':dissolve, 'op_duration_seconds':durations[0],
                            'ed_duration_seconds':durations[2], 'core_start_seconds':offset_one,
                            'source_assets':[{'role':role, 'filename':p.name,
                                              'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                                             for role,p in [('opening', inputs[0]), ('ending', inputs[2])]],
                            'note':'Course-provided assets are not included in the public repository.'}}
    manifest_path.write_text(json.dumps(final, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'video':str(target), 'frames':frames, 'seconds':seconds, 'fps':fps}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opening', required=True)
    parser.add_argument('--ending', required=True)
    args = parser.parse_args()
    compose(args.opening, args.ending)
