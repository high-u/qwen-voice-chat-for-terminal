# Qwen のモデルでターミナル音声チャット

## 概要

- ターミナルアプリ
- Enter キーでユーザープロンプトの音声入力開始
- Enter キーでユーザープロンプトの音声入力終了
- LLM にユーザープロンプトをテキストで渡す
- LLM からの返答を、音声に変換して、音声出力
- 音声出力後、ユーザープロンプト入力用の Enter キーの入力待ち
- マルチターン会話対応

## 技術スタック

- uv
- Python

## 使用モデル

- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8
