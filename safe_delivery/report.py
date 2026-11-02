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
        text(48, 31, f'{identity}  |  MuJoCo 3.3.7  |  2026-11-02', 8.5, GRAY)
        text(width-92, 31, f'{num} / 3', 8.5, GRAY)

    page(1, 'AIで仕分け、滑らかに届ける')
    y = para(724, '目的：搭載済みの荷物をカメラ画像から3色に分類し、棚を避けて正しい搬送先に届ける。認識AIと古典的な経路計画を組み合わせ、搬送成功に加えて動きの滑らかさを検証する。')
    y = heading(y-13, '01', '実装したシステム')
    boxes = [('画像から判断', '色分類MLP'), ('地図から計画', 'A* + 経路短縮'), ('力入力で搬送', '速度・加速度制御')]
    for i, (top, bottom) in enumerate(boxes):
        x = 48+i*171
        c.setFillColor(HexColor('#edf5f6')); c.roundRect(x, y-48, 157, 52, 6, fill=1, stroke=0)
        text(x+12, y-13, top, 11, TEAL); text(x+12, y-34, bottom, 10.5)
        if i<2: text(x+159, y-25, '>', 12, TEAL)
    y -= 75
    y = para(y, 'AI：正規化したRGB値（固定領域の画素中央値）を入力する3→16→3のMLP。合成データ6,000件で教師あり学習し、実際のMuJoCoカメラ画像で推論する。色の正解値を制御器へ直接渡さない。')
    y = para(y-5, '制御：既知の棚配置を機体半径と余裕幅で膨張させ、A*で経路を探索。速度指令の変化を0.55 m/s²以内に制限し、速度サーボから力を入力する。位置は初期化後に書き換えず、MuJoCoで積分する。')
    image_top = y-10
    image_height = 253
    c.drawImage('output/demo_success.png', 48, image_top-image_height, width=499, height=image_height, preserveAspectRatio=True, anchor='c')
    y = image_top-image_height-19
    y = para(y, '図1　評価seed 100の実演。緑の点線は計画経路。右上は搭載カメラ画像。赤い荷物を赤い区画まで搬送した実シミュレーションの最終フレーム。', 9.2, GRAY, 14)
    para(y-8, '対象範囲：平面2自由度の簡略化した全方向移動台車。荷物は車体に固定済み。把持・荷降ろし、車輪接地の詳細モデル、実機実験は含まない。', 9.6, GRAY, 15)
    c.showPage()

    page(2, '評価：成功率と滑らかさを分けて測る')
    y = heading(722, '02', '評価条件を開発時から分離')
    y = para(y, '開発用3配置（seed 10～12）で動作を確認後、制御設定を固定。未使用30配置（seed 100～129、各色10件）を3方式で比較した。開始位置・棚位置・色・照明・車体質量を変化させ、各方式には同一条件を与えた。')
    y = para(y-5, '成功条件：40秒以内に正しい搬送先へ到着し、中心から0.12 m未満、速度0.10 m/s未満、棚との接触なし。接触はMuJoCoの接触情報から判定し、その時点で失敗として終了する。')
    y -= 12
    columns = [48, 218, 315, 420]
    headers = ['方式', '搬送成功', '接触あり', '対応平均時間']
    c.setFillColor(HexColor(NAVY)); c.rect(48, y-10, 499, 28, fill=1, stroke=0)
    for x, h in zip(columns, headers): text(x+9, y, h, 10, '#ffffff')
    for i, (name, key) in enumerate([('直進のみ（参考）', 'direct'), ('A*', 'astar'), ('A* + 加速度制限', 'smooth')]):
        yy = y-37-i*31
        if i==2:
            c.setFillColor(HexColor('#e8f6f1')); c.rect(48, yy-10, 499, 29, fill=1, stroke=0)
        q = summary[key]
        values = [name, f"{q['successes']} / {q['n']}", f"{q['collisions']} / {q['n']}",
                  '-' if key=='direct' else f"{q['paired_mean_sim_time']:.2f} s"]
        for x, value in zip(columns, values): text(x+9, yy, value, 10)
    y -= 142
    y = para(y, '直進は障害物回避を持たない弱い参考方式。主要な比較は同じA*経路を使う2方式であり、両方成功した30配置を対応させて比較した。直進失敗例の短い終了時間を搬送時間と混同しない。', 9.6, GRAY, 15)
    y = heading(y-15, '03', '主結果：滑らかさと速度のトレードオフ')
    for name, key, color in [('A*', 'astar', '#6a7d91'), ('A* + 加速度制限', 'smooth', TEAL)]:
        value = summary[key]['paired_mean_jerk_rms_m_s3']
        text(48, y, name, 10)
        c.setFillColor(HexColor(color)); c.roundRect(198, y-4, value*60, 14, 3, fill=1, stroke=0)
        text(448, y, f'{value:.3f} m/s³', 10, color)
        y -= 31
    y = para(y-2, f"図2　各走行のジャークRMSの平均。0.05秒間隔の実速度から2階差分で算出。提案方式で{jerk_reduction:.1f}%減少した。一方、時間は{a['paired_mean_sim_time']:.2f}秒から{s['paired_mean_sim_time']:.2f}秒へ増加し、経路長も{a['paired_mean_path_m']:.2f} mから{s['paired_mean_path_m']:.2f} mへ増加した。", 10, NAVY, 16)
    y = para(y-8, f"認識結果：独立な合成データ{training['test_n']:,}件で{training['test_accuracy']*100:.1f}%、評価時の実レンダリング画像では{recognition}/30件が正しい色に分類された。灰色は受理条件を満たさず、走行開始を拒否することを別テストで確認した。", 10, NAVY, 16)
    para(y-8, '解釈上の限界：30/30は一般環境での100%成功を意味しない。独立ベルヌーイ試行を仮定したWilson 95%区間は約88.6～100%。この評価は同じ棚構造の摂動に限定される。ジャークは荷物の破損を直接測る指標ではない。', 9.5, GRAY, 15)
    c.showPage()

    page(3, '工夫・再現方法・残る課題')
    y = heading(722, '04', 'これまでの経験を活かした工夫')
    y = para(y, '競技で局所的な成功率と提出スコアの差を経験したことから、少数の成功映像だけで評価しない方針にした。開発と評価の配置を分け、全90走行の成否・接触・時間・軌跡を保存する。動画は事前に決めた評価先頭3配置を使用し、記録済みの結果と一致することを確認した。')
    y = para(y-5, 'AIの役割を色分類に絞り、経路と低レベル制御を分離したため、失敗原因を追いやすい。未知色を無理に3色へ振り分けないため、出力確率0.80、彩度0.30、明るさ0.08の受理条件を併用する。ただし一般的な未知物体検出ではない。')
    y = heading(y-15, '05', '再現方法と提出物')
    y = para(y, 'Python 3.12、MuJoCo 3.3.7、NumPy 2.3.4。学習・制御・物理演算はCPUで実行。macOS arm64で検証した。描画にはOpenGLが必要。フォント・OS・数値ライブラリにより描画や数値に小差が生じうる。')
    for line in ['python -m pip install -r requirements.txt',
                 'python -m unittest discover -s tests -v',
                 'python -m safe_delivery.simulation --split heldout',
                 'python -m safe_delivery.media',
                 'python -m safe_delivery.report']:
        text(60, y-3, line, 10, TEAL); y -= 20
    y = para(y-4, '学習の再実行：python -m safe_delivery.perception。学習済み重み、乱数seed、評価プロトコル、manifestのSHA-256、各走行JSON/NPZ、テストを同梱する。動画は約92秒・1280×720・20fps・字幕付き。走行映像は1倍速で、結果確認の静止表示を明記した。', 10, NAVY, 16)
    y = heading(y-15, '06', '制約と次の改善')
    y = para(y, '固定された荷物領域の色分類であり、一般物体検出ではない。地図・自己位置は既知の理想値で、SLAMやセンサ誤差は扱わない。棚は静止、経路は開始時に計画し、歩行者に対する再計画は未実装。物理モデルは平面移動に制約し、実際の車輪駆動を再現していない。', 10, NAVY, 16)
    y = para(y-5, '次の段階では、未知の棚構造・動く障害物・画像上の荷物位置の変化を含めて評価し、必要な改善を一つずつ加える。制御パラメータの最適性や実機安全性は主張しない。', 10, NAVY, 16)
    y = heading(y-15, '07', '参照・制作について')
    y = para(y, '課題要件：Physical AI 応用1「最終課題（10/19改訂版）」1～4頁。実装は課題サンプルのコピーではなく、本課題向けに新規作成。生成AI（Codex）を設計・実装・資料作成に使用し、記載した実測値は同梱の実行記録から集計した。', 9.5, GRAY, 14)
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
