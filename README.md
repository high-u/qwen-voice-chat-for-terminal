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

インストール例

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

```bash
python main.py
```

1. Enter キー押下で、ユーザープロンプトの録音開始。
2. Enter キー押下で、ユーザープロンプトの録音終了。

## それぞれの検証

```bash
# GPU モード
python test_asr.py --mode gpu
python test_tts.py --mode gpu

# CPU オフロードモード
python test_asr.py --mode cpu-offload
python test_tts.py --mode cpu-offload
```
