"""Re-render real rollouts and assemble a captioned 1x demonstration video."""
from pathlib import Path
import json
import os
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from .simulation import make_scene, run, setup, observe
from .perception import Classifier, LABELS
from .settings import account_id

FPS, W, H = 20, 1280, 720
NAVY, WHITE, TEAL, MUTED = '#101f30', '#f2f6fa', '#4ee0ba', '#acbdcc'


def font(size):
    candidates = [os.environ.get('PAI_JAPANESE_FONT', ''),
                  '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
                  '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for p in candidates:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError('Set PAI_JAPANESE_FONT to a Japanese font file; no font is bundled.')


def text(draw, xy, value, size=26, color=WHITE):
    draw.text(xy, value, font=font(size), fill=color)


def card(title, lines, eyebrow='PAI / APPLIED FINAL PROJECT'):
    im = Image.new('RGB', (W, H), NAVY)
    d = ImageDraw.Draw(im)
    d.rectangle((68, 74, 128, 80), fill=TEAL)
    text(d, (68, 102), eyebrow, 21, TEAL)
    text(d, (68, 180), title, 43)
    for i, line in enumerate(lines):
        text(d, (72, 286+i*62), line, 29, WHITE if i==0 else MUTED)
    text(d, (72, 650), 'AI認識 × 経路計画 × MuJoCo物理シミュレーション', 20, MUTED)
    return im


def add_route(renderer, route):
    for a, b in zip(route[:-1], route[1:]):
        for p in np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/.12))):
            scn = renderer.scene
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([.022, .022, .022]), np.array([p[0], p[1], .035]),
                               np.eye(3).ravel(), np.array([.08, .72, .55, 1.]))
            scn.ngeom += 1


class Recorder:
    def __init__(self, writer, title, seed):
        self.writer, self.title, self.seed = writer, title, seed
        self.renderer = None
        self.last, self.frames = None, 0
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, 0, .1]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 10.2, 90, -72

    def __call__(self, m, d, route, perception, collision, cargo_image, append=True):
        self.arguments = (m, d, route, perception, collision, cargo_image)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, height=720, width=960)
        self.renderer.update_scene(d, camera=self.camera)
        add_route(self.renderer, route)
        im = Image.new('RGB', (W, H), NAVY)
        im.paste(Image.fromarray(self.renderer.render()), (0, 0))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, 960, 76), fill=NAVY)
        text(draw, (25, 16), self.title, 29)
        draw.rectangle((0, 648, 960, 720), fill=NAVY)
        text(draw, (27, 667), '実シミュレーション / 1倍速 / 緑の点線 = 計画経路', 22, WHITE)
        text(draw, (984, 26), 'CARGO CAMERA', 22, TEAL)
        im.paste(Image.fromarray(cargo_image).resize((160, 160)), (1040, 78))
        label = LABELS[perception['label']] if perception['label'] is not None else 'REJECT'
        text(draw, (984, 258), f'AI分類: {label}', 26)
        text(draw, (984, 306), f"出力確率: {perception['confidence']:.3f}", 21, MUTED)
        text(draw, (984, 370), f't = {d.time:05.2f} s', 29)
        text(draw, (984, 424), f'速度 {np.linalg.norm(d.qvel):.2f} m/s', 23, MUTED)
        text(draw, (984, 480), f'配置 seed {self.seed}', 22, MUTED)
        text(draw, (984, 554), '棚に接触 / 終了' if collision else '棚との接触なし', 23, '#ff8e88' if collision else TEAL)
        text(draw, (984, 645), '固定カメラ画像で認識', 18, MUTED)
        if append:
            self.writer.append_data(np.asarray(im))
            self.frames += 1
        self.last = im

    def close(self):
        if self.renderer:
            self.renderer.close()


def build():
    out = Path('output')
    out.mkdir(exist_ok=True)
    summary = json.loads(Path('results/heldout/summary.json').read_text())
    rows = json.loads(Path('results/heldout/episodes.json').read_text())
    model = Classifier('models/color_mlp.npz')
    identity = account_id()
    video = out / f'PAI最終課題_{identity}.mp4'
    writer = imageio.get_writer(str(video), fps=FPS, codec='libx264', quality=8,
                                macro_block_size=16, ffmpeg_params=['-movflags', '+faststart'])
    chapters, count, demos = [], 0, []

    def still(im, seconds, title):
        nonlocal count
        chapters.append({'title': title, 'start_seconds': count/FPS, 'duration_seconds': seconds})
        for _ in range(round(seconds*FPS)):
            writer.append_data(np.asarray(im))
            count += 1

    still(card('AIで仕分け、滑らかに届ける', [
        '色を認識して、棚を避けながら正しい搬送先へ。',
        '学習済みMLPと物理シミュレータを接続した搬送ロボット',
        f'PAI応用1 最終課題 / {identity}']), 7, '課題と目的')
    still(card('小さなAIを、確かに動くシステムへ', [
        'カメラ画像 → 色分類MLP → 搬送先の決定',
        '既知の地図 → A*経路計画 → 加速度を制限した速度指令',
        '力入力で走行。座標の書き換えによる動画ではありません。',
        '荷物は搭載済み。把持・荷降ろしは対象外。']), 9, '構成')

    for method, seed, label, title in [
        ('direct', 100, 0, '比較：直進のみでは棚に衝突'),
        ('smooth', 100, 0, '実演 1：赤い荷物を赤い搬送先へ'),
        ('smooth', 101, 1, '実演 2：緑の荷物 / 別の配置・照明'),
        ('smooth', 102, 2, '実演 3：青い荷物 / 別の配置・質量')]:
        rec = Recorder(writer, title, seed)
        result, _ = run(make_scene(seed, label), method, model, rec)
        reference = next(x for x in rows if x['seed']==seed and x['method']==method)
        assert result == reference, 'Video rollout differs from recorded evaluation'
        demos.append(result)
        chapters.append({'title': title, 'start_seconds': count/FPS,
                         'duration_seconds': rec.frames/FPS, 'sim_time': result['sim_time'], 'speed': '1x'})
        count += rec.frames
        if rec.last is not None:
            # Render the exact final physics state for the labelled result still.
            # The regular movie remains sampled at 20 fps with no extra frames.
            rec(*rec.arguments[:4], result['collision'], rec.arguments[5], append=False)
            last = rec.last.copy()
            draw = ImageDraw.Draw(last)
            draw.rounded_rectangle((180, 530, 800, 625), radius=12, fill=NAVY)
            text(draw, (216, 547), '搬送成功：正しい色の搬送先に到着' if result['success'] else '接触を検知して終了', 29, TEAL)
            text(draw, (216, 588), '結果確認のため静止表示', 19, MUTED)
            still(last, 3, '結果静止表示')
            if method=='smooth' and seed==100:
                last.save(out/'demo_success.png')
                Image.fromarray(observe(*setup(make_scene(100, 0)))).save(out/'cargo_camera.png')
        rec.close()

    gray = make_scene(10, 0)
    gray.cargo_rgb = [.5, .5, .5]
    m, d = setup(gray)
    camera_image = observe(m, d)
    pred = model.predict(camera_image)
    assert pred['label'] is None
    im = card('未知の色は、無理に仕分けない', [
        '灰色の荷物 → 受理条件を満たさず、走行を開始しない。',
        '出力確率だけでなく、彩度・明るさも確認。',
        'すべての未知物体を検知できる保証はありません。'])
    im.paste(Image.fromarray(camera_image).resize((160, 160)), (1020, 478))
    still(im, 7, '未知色の拒否')

    a, s = summary['astar'], summary['smooth']
    reduction = (1-s['paired_mean_jerk_rms_m_s3']/a['paired_mean_jerk_rms_m_s3'])*100
    extra = s['paired_mean_sim_time']-a['paired_mean_sim_time']
    still(card('開発に使っていない30配置で比較', [
        f"直進のみ：成功 {summary['direct']['successes']}/30　棚との接触 {summary['direct']['collisions']}/30",
        f"A*：成功 {a['successes']}/30　提案方式：成功 {s['successes']}/30",
        f'提案方式は平均ジャークRMSを {reduction:.1f}% 低減。',
        f'対価は平均 {extra:.2f} 秒の所要時間増加（同一30配置比較）。']), 10, '実測結果')
    still(card('過去の反省を、評価設計に活かす', [
        '開発3配置と評価30配置を分離し、失敗も保存。',
        '成功率に加え、衝突・時間・滑らかさを記録。',
        '既知の静的地図・理想位置情報・3色の分類に限定。',
        '次の課題：実機、動的障害物、物体検出、荷降ろし。']), 9, '工夫と限界')
    writer.close()
    manifest = {'fps': FPS, 'frames': count, 'duration_seconds': count/FPS,
                'resolution': [W, H], 'audio': 'none; Japanese captions',
                'playback': 'Simulation segments 1x, plus clearly marked stills',
                'chapters': chapters, 'demos': demos, 'gray_prediction': pred}
    (out/'video_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    assert 60 <= count/FPS <= 180
    print(json.dumps({'video': str(video), 'seconds': count/FPS, 'frames': count}, ensure_ascii=False))


if __name__ == '__main__':
    build()
