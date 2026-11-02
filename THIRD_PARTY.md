# 外部依存と出典

- MuJoCo 3.3.7：物理演算・描画。[公式リポジトリ](https://github.com/google-deepmind/mujoco)、[Apache-2.0ライセンス](https://github.com/google-deepmind/mujoco/blob/main/LICENSE)。ライブラリはPyPIから導入し、ソースやバイナリを本提出ZIPへ転載していません。
- NumPy：数値演算。Pillow：字幕・画像処理。imageio / imageio-ffmpeg：動画エンコード。ReportLab：PDF生成。pypdf：検証。各依存物の利用条件は、それぞれの配布物のライセンスに従います。
- 動画の日本語フォントは実行環境のヒラギノ、PDFの日本語は実行環境のArial Unicodeを使用しました。フォントファイルは同梱していません。PDFには使用グリフのサブセットを埋め込んでいます。
- ロボット・棚・床・荷物・色分類用の学習データはコードで生成しています。外部の学習済みモデル・画像・動画・音声は使用していません。
- 配布された課題PDF、課題サンプル、講座ジングルは公開用ソースZIPに含めていません。
- 生成AI（Codex）を制作支援に使用しました。アルゴリズムの学術的新規性や、外部の競技成績は主張しません。
