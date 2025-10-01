# AIS2 v2 - Multi-Object Tracking System

RT-DETRv2を使用した物体検出・追跡システム

## 概要

このプロジェクトは、RT-DETRv2の軽量モデルを使用して動画内の物体（特に馬）を検出・追跡するシステムです。

## 特徴

- RT-DETRv2の軽量モデルを使用した高速物体検出
- MacBook M2に最適化された軽量実装
- 動画ファイルからの物体検出・追跡
- バウンディングボックス付きの結果動画出力

## 必要環境

- Python 3.8+
- MacBook M2 (Apple Silicon)
- uv (パッケージマネージャー)

## インストール

```bash
# 仮想環境の作成
uv venv

# 仮想環境の有効化
source .venv/bin/activate

# 依存関係のインストール
uv pip install -e .
```

## 使用方法

```bash
python src/main.py --input videos/input.mp4 --output videos/output.mp4
```

## プロジェクト構造

```
AIS2_v2/
├── src/
│   ├── mot_system/
│   │   ├── detection/
│   │   ├── tracking/
│   │   └── utils/
│   └── main.py
├── models/
├── videos/
├── tests/
└── pyproject.toml
```
