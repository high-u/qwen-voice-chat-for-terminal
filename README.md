# Qwen Voice Chat for Terminal

## 動作確認環境

| 項目 | 内容 |
|:---|:---|
| OS | Ubuntu 24.04 |
| GPU | NVIDIA RTX 5070 Ti |
| CUDA | 13.0 |
| Python | 3.13 |

### システム依存関係

- PortAudio (`libportaudio2`): マイク録音およびスピーカー再生に使用
- FFmpeg: 音声データのデコードおよびモデル入力用の前処理に使用
- SoX: 音声の正規化や変換プロセスに使用

Ubuntu でのインストール例

```bash
sudo apt update && sudo apt install libportaudio2 ffmpeg sox
```

## 環境構築

```bash
git clone https://github.com/high-u/qwen-voice-chat-for-terminal.git
cd qwen-voice-chat-for-terminal
uv sync
source .venv/bin/activate
```

## 使用方法

### 1 音声プレセット作成（マイクで録音）

実行すると:

1. モデルをロード
2. "Press Enter to start recording..." と表示される
3. Enter を押すと録音開始
4. もう一度 Enter を押すと録音停止
5. プリセットを保存（voice_presets/my_voice.pkl）
6. Ctrl+C で終了

```
python voice_clone.py create --name my_voice --text "録音する内容のテキスト"
```

プリセットで再生テスト

```
python voice_clone.py synthesize --name my_voice --text "こんにちは！今日は良い天気ですね！"
```

### 2 ボイスチャット

実行すると:

1. モデルをロード
2. "Press Enter to start recording..." と表示される
3. Enter を押すとユーザープロンプトの録音開始
4. もう一度 Enter を押すと録音停止
5. 音声 → テキスト（ユーザープロンプト） → LLM → テキスト（LLM 回答） → 音声
6. 引き続き会話を続ける場合は、Enter を押すとユーザープロンプトの録音開始
7. Ctrl+C で終了

```bash
python main.py --voice my_voice
```

## TTS・ASR の検証

```bash
# GPU モード
python test_asr.py --mode gpu
python test_tts.py --mode gpu --voice my_voice

# CPU オフロードモード
python test_asr.py --mode cpu-offload
python test_tts.py --mode cpu-offload --voice my_voice
```

## TTS 計測

```bash
python measure_tts.py -t "こんにちは、これはテストです。" -v my_voice --mode gpu
python measure_tts.py -t "こんにちは、これはテストです。" -v my_voice --mode cpu-offload
```
