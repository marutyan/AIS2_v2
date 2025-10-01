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

### 基本的な使用方法

```bash
# 仮想環境を有効化
source .venv/bin/activate

# 馬の検出・追跡を実行
python src/main.py --input videos/realhorses.mp4 --target-class horse --output outputs/result.mp4

# リアルタイム表示付きで実行
python src/main.py --input videos/realhorses.mp4 --target-class horse --display

# 信頼度を調整して実行
python src/main.py --input videos/realhorses.mp4 --target-class horse --conf 0.3 --output outputs/result.mp4
```

### コマンドラインオプション

- `--input, -i`: 入力動画ファイルのパス（必須）
- `--output, -o`: 出力動画ファイルのパス（省略時は自動生成）
- `--target-class, -t`: 追跡対象のクラス名（デフォルト: horse）
- `--conf, -c`: 信頼度の閾値（デフォルト: 0.5）
- `--display, -d`: リアルタイム表示を有効化
- `--device`: 使用するデバイス（cpu, cuda, mps）

### テスト実行

```bash
# システムテストを実行
python test_system.py
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
