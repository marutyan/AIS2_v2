#!/usr/bin/env python3
"""
データセットアップスクリプト
AIS2の動画ファイルをコピーして使用する
"""

import shutil
from pathlib import Path
import os


def setup_sample_video():
    """サンプル動画をセットアップ"""
    
    # AIS2の動画パスを確認
    ais2_video_path = Path("/Users/marutyan/CVLAB/AIS2/videos/realhorses.mp4")
    
    # 動画ディレクトリを作成
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    
    # 動画をコピー
    target_path = videos_dir / "realhorses.mp4"
    
    if ais2_video_path.exists():
        print(f"動画をコピー中: {ais2_video_path} -> {target_path}")
        shutil.copy2(ais2_video_path, target_path)
        print("動画のコピーが完了しました。")
        return str(target_path)
    else:
        print(f"警告: 元の動画ファイルが見つかりません: {ais2_video_path}")
        print("手動で動画ファイルを videos/ ディレクトリに配置してください。")
        return None


def create_gitignore():
    """適切な.gitignoreファイルを作成"""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
models/*.pt
models/*.pth
models/*.onnx
videos/*.mp4
videos/*.avi
videos/*.mov
outputs/
logs/
temp/

# But keep example files
!videos/README.md
!models/README.md
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print(".gitignoreファイルを作成しました。")


def create_readme_files():
    """各ディレクトリにREADMEファイルを作成"""
    
    # videos/README.md
    videos_readme = """# Videos Directory

このディレクトリには動画ファイルを配置します。

## サポートされる形式
- MP4
- AVI
- MOV

## 使用方法
```bash
python src/main.py --input videos/your_video.mp4
```
"""
    
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    with open(videos_dir / "README.md", "w") as f:
        f.write(videos_readme)
    
    # models/README.md
    models_readme = """# Models Directory

このディレクトリにはモデルファイルを配置します。

RT-DETRv2のモデルは自動的にダウンロードされます。

## モデルファイル
- RT-DETRv2の軽量モデル（自動ダウンロード）
- カスタムモデル（オプション）
"""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "README.md", "w") as f:
        f.write(models_readme)
    
    print("READMEファイルを作成しました。")


def main():
    """メイン関数"""
    print("データセットアップを開始します...")
    
    # 必要なディレクトリを作成
    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # ファイルを作成
    create_gitignore()
    create_readme_files()
    
    # サンプル動画をセットアップ
    video_path = setup_sample_video()
    
    print("\nセットアップが完了しました！")
    
    if video_path:
        print(f"\nテスト実行コマンド:")
        print(f"python src/main.py --input {video_path} --target-class horse --display")
    
    print("\nGitコマンド:")
    print("git add .")
    print("git commit -m 'Initial setup with RT-DETRv2 MOT system'")
    print("git push origin main")


if __name__ == "__main__":
    main()
