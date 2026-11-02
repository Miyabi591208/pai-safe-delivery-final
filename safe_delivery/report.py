"""Build the submission PDF from recorded evidence, not hand-entered scores."""
from pathlib import Path
import json
import os
import re
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from .settings import account_id

FONT = 'EmbeddedJapanese'
NAVY, TEAL, GRAY = '#14273a', '#087e75', '#526476'


def build():
    font_path = os.environ.get('PAI_PDF_FONT', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not Path(font_path).exists():
        raise RuntimeError('Set PAI_PDF_FONT to a Japanese TrueType font. The font must permit PDF embedding.')
    pdfmetrics.registerFont(TTFont(FONT, font_path))
    out = Path('output/pdf')
    out.mkdir(parents=True, exist_ok=True)
    identity = account_id()
    path = out/f'説明資料_PAI最終課題_{identity}.pdf'
    c = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    c.setTitle('AIで仕分け、滑らかに届ける - PAI最終課題')
    c.setAuthor(identity)
    c.setSubject('学習済み色分類器と経路計画による搬送ロボットの実装・検証')
    width, height = A4
    rows = json.loads(Path('results/heldout/episodes.json').read_text())
    summary = json.loads(Path('results/heldout/summary.json').read_text())
    training = json.loads(Path('models/training.json').read_text())
    video = json.loads(Path('output/video_manifest.json').read_text())
    a, s = summary['astar'], summary['smooth']
    jerk_reduction = (1-s['paired_mean_jerk_rms_m_s3']/a['paired_mean_jerk_rms_m_s3'])*100
    recognition = sum(r['perception']['label']==r['true_label'] for r in rows if r['method']=='smooth')

    def text(x, y, value, size=10.5, color=NAVY):
        c.setFillColor(HexColor(color)); c.setFont(FONT, size); c.drawString(x, y, value)

    def para(y, value, size=10.5, color=NAVY, leading=17, x=48, maxwidth=499):
        line = ''
        # Keep ASCII words/numbers intact and avoid orphaned Japanese punctuation.
        for char in re.findall(r'[A-Za-z0-9][A-Za-z0-9_./%:²³+*～-]*|[^\n]|\n', value):
            overflow = pdfmetrics.stringWidth(line+char, FONT, size)>maxwidth
            if char in '。、，．！？）】」』' and line:
                overflow = False
            if char=='\n' or overflow:
                text(x, y, line, size, color); y -= leading; line = ''
            if char!='\n':
                line += char if line else char.lstrip()
        if line:
            text(x, y, line, size, color); y -= leading
        return y

    def heading(y, number, value):
        text(48, y, number, 13, TEAL)
        text(79, y, value, 15)
        return y-27

    def page(num, section):
        c.setFillColor(HexColor(NAVY)); c.rect(0, height-91, width, 91, fill=1, stroke=0)
        text(46, height-32, 'PHYSICAL AI / APPLIED FINAL PROJECT', 10, '#7be1ca')
        text(46, height-64, section, 22, '#ffffff')
        c.setStrokeColor(HexColor('#d9e2e8')); c.line(48, 48, width-48, 48)
        text(48, 31, f'{identity}  |  MuJoCo 3.3.7  |  2026-11-03', 8.5, GRAY)
        text(width-92, 31, f'{num} / 3', 8.5, GRAY)

    page(1, 'AIで仕分け、滑らかに届ける')
    y = para(724, 'カメラで荷物の色を見分け、棚を避けて対応する搬送先に届ける。本作品では、目的地に着くことに加え、動きの急な変化を抑える制御と、その効果を確かめる評価方法を工夫した。')
    c.setFillColor(HexColor('#e8f6f1')); c.roundRect(48, y-87, 499, 77, 6, fill=1, stroke=0)
    text(60, y-30, '一番の工夫：同じ計画経路でも、速度の変え方を滑らかにする', 12, TEAL)
    text(60, y-50, f'加速度制限を追加し、平均ジャークRMSを{jerk_reduction:.1f}%低減。', 11)
    text(60, y-70, f"両方式とも30/30件で搬送成功。所要時間は平均{s['paired_mean_sim_time']-a['paired_mean_sim_time']:.2f}秒増加。", 10)
    y -= 100
    y = heading(y-13, '01', '実装したシステム')
    boxes = [('画像から判断', '色分類MLP'), ('地図から計画', 'A* + 経路短縮'), ('力入力で搬送', '速度・加速度制御')]
    for i, (top, bottom) in enumerate(boxes):
        x = 48+i*171
        c.setFillColor(HexColor('#edf5f6')); c.roundRect(x, y-48, 157, 52, 6, fill=1, stroke=0)
        text(x+12, y-13, top, 11, TEAL); text(x+12, y-34, bottom, 10.5)
        if i<2: text(x+159, y-25, '>', 12, TEAL)
    y -= 75
    y = para(y, '認識：小型ニューラルネットワーク（MLP）を合成RGBデータ6,000件で学習。カメラ画像の荷物領域から赤・緑・青を推定し、搬送先を選ぶ。出力確率・彩度・明るさの基準を一つでも満たさなければ、搬送を開始しない。')
    y = para(y-5, '移動：棚との間に機体の大きさを考慮した余裕を設け、A*で経路を計画する。速度指令の変化を0.55 m/s²以内に制限し、力を加えて台車を動かす。位置を直接書き換えず、MuJoCoの物理演算で移動させる。')
    image_top = y-10
    image_height = 210
    c.drawImage('output/demo_success.png', 48, image_top-image_height, width=499, height=image_height, preserveAspectRatio=True, anchor='c')
    y = image_top-image_height-19
    y = para(y, '図1　評価seed 100の実演。緑の点線は計画経路。右上は搭載カメラ画像。赤い荷物を赤い区画まで搬送した実シミュレーションの最終フレーム。', 9.2, GRAY, 14)
    y = para(y-8, '既存の大規模モデルの追加学習ではなく、色分類AI・経路計画・制御を本課題向けに組み合わせた。対象は平面移動の簡略化した台車で、荷物は固定済み。把持・荷降ろし・実機実験は含まない。', 9.6, GRAY, 15)
    if y < 55:
        raise RuntimeError(f'Page 1 overflows: y={y}')
    c.showPage()

    page(2, '評価：成功率と滑らかさを分けて測る')
    y = heading(722, '02', '同じ条件で、加速度制限の効果を比べる')
    y = para(y, '開発用3配置（seed 10～12）で動作を確認後、制御設定を固定。未使用30配置（seed 100～129、各色10件）を3方式で比較した。開始位置・棚位置・色・照明・車体質量を変化させ、各方式には同一条件を与えた。')
    y = para(y-5, '成功条件：40秒以内に正しい搬送先へ到着し、中心から0.12 m未満、速度0.10 m/s未満、棚との接触なし。接触はMuJoCoの接触情報から判定し、その時点で失敗として終了する。')
    y -= 12
    columns = [48, 218, 315, 420]
    headers = ['方式', '搬送成功', '接触あり', '平均搬送時間']
    c.setFillColor(HexColor(NAVY)); c.rect(48, y-10, 499, 28, fill=1, stroke=0)
    for x, h in zip(columns, headers): text(x+9, y, h, 10, '#ffffff')
    for i, (name, key) in enumerate([('直進のみ（参考）', 'direct'), ('A* + 速度制御（比較）', 'astar'), ('同方式 + 加速度制限（提案）', 'smooth')]):
        yy = y-37-i*31
        if i==2:
            c.setFillColor(HexColor('#e8f6f1')); c.rect(48, yy-10, 499, 29, fill=1, stroke=0)
        q = summary[key]
        values = [name, f"{q['successes']} / {q['n']}", f"{q['collisions']} / {q['n']}",
                  '-' if key=='direct' else f"{q['paired_mean_sim_time']:.2f} s"]
        for x, value in zip(columns, values): text(x+9, yy, value, 10)
    y -= 142
    y = para(y, '比較方式と提案方式では、色分類AI・計画経路・速度制御を共通にし、速度指令の加速度制限の有無だけを変えた。両方成功した同じ30配置で時間と滑らかさを比較した。直進は障害物回避の必要性を示す参考例である。', 9.6, GRAY, 15)
    y = heading(y-15, '03', '搬送成功を保ち、動きの急な変化を抑えた')
    y = para(y, 'ジャークは「加速度がどれだけ急に変わるか」を表す。走行ごとにその大きさをRMS（二乗平均平方根）で求め、30走行の平均を比べた。値が小さいほど、加速度の変化が穏やかである。', 10, NAVY, 16)
    y -= 8
    for name, key, color in [('比較：加速度制限なし', 'astar', '#6a7d91'), ('提案：加速度制限あり', 'smooth', TEAL)]:
        value = summary[key]['paired_mean_jerk_rms_m_s3']
        text(48, y, name, 10)
        c.setFillColor(HexColor(color)); c.roundRect(198, y-4, value*60, 14, 3, fill=1, stroke=0)
        text(448, y, f'{value:.3f} m/s³', 10, color)
        y -= 31
    y = para(y-2, f"図2　平均ジャークRMSは{jerk_reduction:.1f}%減少。所要時間は{a['paired_mean_sim_time']:.2f}秒から{s['paired_mean_sim_time']:.2f}秒へ、実際の走行距離は{a['paired_mean_path_m']:.2f} mから{s['paired_mean_path_m']:.2f} mへ増加した。速度と滑らかさの両面を示し、今回の改善を評価した。指標は0.05秒ごとの実速度の2階差分から算出した。", 10, NAVY, 16)
    y = para(y-8, f"認識結果：独立な合成データ{training['test_n']:,}件で{training['test_accuracy']*100:.1f}%、評価時の実レンダリング画像では{recognition}/30件が正しい色に分類された。灰色は受理条件を満たさず、走行開始を拒否することを別テストで確認した。", 10, NAVY, 16)
    y = para(y-8, '評価の範囲：30/30は一般環境での100%成功を意味しない。独立試行を仮定した成功率のWilson 95%区間は約88.6～100%。同じ棚構造で位置などを変えた評価に限られ、荷物の破損や実機の安全性は測っていない。', 9.5, GRAY, 15)
    if y < 55:
        raise RuntimeError(f'Page 2 overflows: y={y}')
    c.showPage()

    page(3, '工夫・再現方法・残る課題')
    y = heading(722, '04', '何を工夫し、どこまで確かめたか')
    for title, detail in [
        ('1. 動きの急な変化を抑える', f'A*と速度制御に加速度制限を追加した。同じ30配置で搬送成功を保ち、平均ジャークRMSを{jerk_reduction:.1f}%低減した。今回、比較実験で効果を確認した中心的な工夫である。'),
        ('2. 判断が難しいときは動かさない', '出力確率0.80・彩度0.30・明るさ0.08の基準をすべて満たす場合だけ搬送する。灰色と暗い画像の拒否をテストした。あらゆる未知色を検出できるとは限らない。'),
        ('3. 学習時から見え方に変化をつける', '明るさ・色味・ノイズを変えた6,000件で色分類AIを学習した。評価用のカメラ画像30件で色を正しく分類した。ただし、データの変化を加えない学習との比較は行っていない。'),
        ('4. 成功例だけで性能を判断しない', '競技で手元の評価と提出結果の差を経験し、開発3配置と評価30配置を分離した。全90走行を記録し、動画の3配置も事前に決めた。成功率だけでなく、接触・時間・滑らかさを示した。')]:
        text(48, y, title, 11, TEAL); y -= 19
        y = para(y, detail, 10, NAVY, 15)-8
    y = heading(y-15, '05', '再現方法と提出物')
    y = para(y, 'Python 3.12、MuJoCo 3.3.7、NumPy 2.3.4を使用し、macOS arm64で検証。学習・制御・物理演算はCPUで実行し、描画にはOpenGLを使用する。ソース、重み、評価条件、全走行の記録、9項目のテストを同梱した。', 10, NAVY, 15)
    y = para(y-4, 'READMEに環境構築・再学習・評価・動画とPDFの再生成手順を記載した。評価の実行コマンド：', 10, NAVY, 15)
    text(60, y-3, 'python -m safe_delivery.simulation --split heldout', 10, TEAL); y -= 23
    y = para(y, f"動画は約{video['duration_seconds']:.0f}秒・1280×720・{video['fps']}fps。走行は1倍速で、結果確認の静止表示を明記した。", 10, NAVY, 15)
    if 'composition' in video:
        y = para(y, '講座配布のOP・EDを0.4秒のフェードで接続。音声はなく日本語字幕で説明する。配布素材自体は公開ソースに含めない。', 10, NAVY, 15)
    y = heading(y-15, '06', '制約と次の改善')
    y = para(y, '荷物の画像上の位置は固定し、地図・自己位置は既知とする。実際の車輪駆動、動く障害物への再計画、実機での動作は未検証である。次は棚構造・障害物の動き・荷物の見える位置を変え、適用範囲を確かめる。', 10, NAVY, 15)
    y = heading(y-15, '07', '参照・制作について')
    y = para(y, '課題要件：Physical AI 応用1「最終課題（10/19改訂版）」1～4頁。本課題向けに新規実装し、生成AI（Codex）を設計・実装・資料作成に使用した。既存手法を目的に合わせて組み合わせた作品であり、新しい学習アルゴリズムの提案ではない。', 9.5, GRAY, 14)
    for title, url in [('MuJoCo公式Python文書', 'https://mujoco.readthedocs.io/en/3.3.7/python.html'),
                       ('MuJoCo公式MJCF仕様', 'https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html')]:
        text(48, y, title, 9, TEAL); text(48, y-13, url, 8.5, GRAY)
        c.linkURL(url, (48, y-17, 545, y+10), relative=0)
        y -= 32
    if y < 55:
        raise RuntimeError(f'Page 3 overflows: y={y}')
    c.save()
    print(path)


if __name__ == '__main__':
    build()
