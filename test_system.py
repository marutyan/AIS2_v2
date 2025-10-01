#!/usr/bin/env python3
"""
システムテストスクリプト
RT-DETRv2 MOTシステムの動作確認
"""

import sys
import cv2
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mot_system.detection.rt_detr_detector import RTDETRv2Detector
from src.mot_system.tracking.simple_tracker import SimpleTracker, draw_tracks
from src.mot_system.utils.video_processor import VideoProcessor


def test_detector():
    """検出器のテスト"""
    print("=== 検出器テスト ===")
    
    try:
        detector = RTDETRv2Detector(conf_threshold=0.3, device="cpu")
        print("✓ 検出器の初期化に成功")
        
        # テスト画像を作成（ダミー）
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # 単色で塗りつぶし
        
        # 検出実行
        detections = detector.detect(test_image)
        print(f"✓ 検出実行完了（検出数: {len(detections)}）")
        
        return True
        
    except Exception as e:
        print(f"✗ 検出器テスト失敗: {e}")
        return False


def test_tracker():
    """トラッカーのテスト"""
    print("\n=== トラッカーテスト ===")
    
    try:
        tracker = SimpleTracker()
        
        # ダミー検出結果
        dummy_detections = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.8,
                'class_id': 17,  # horse
                'class_name': 'horse'
            }
        ]
        
        tracks = tracker.update(dummy_detections)
        print(f"✓ トラッカー動作確認完了（トラック数: {len(tracks)}）")
        
        return True
        
    except Exception as e:
        print(f"✗ トラッカーテスト失敗: {e}")
        return False


def test_video_processor():
    """動画処理のテスト"""
    print("\n=== 動画処理テスト ===")
    
    video_path = "videos/realhorses.mp4"
    
    if not Path(video_path).exists():
        print(f"✗ テスト動画が見つかりません: {video_path}")
        return False
    
    try:
        with VideoProcessor(video_path) as processor:
            print(f"✓ 動画読み込み成功")
            print(f"  - サイズ: {processor.width}x{processor.height}")
            print(f"  - FPS: {processor.fps}")
            print(f"  - フレーム数: {processor.total_frames}")
            print(f"  - 長さ: {processor.duration:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"✗ 動画処理テスト失敗: {e}")
        return False


def test_integration():
    """統合テスト（最初の数フレームのみ）"""
    print("\n=== 統合テスト ===")
    
    video_path = "videos/realhorses.mp4"
    
    if not Path(video_path).exists():
        print(f"✗ テスト動画が見つかりません: {video_path}")
        return False
    
    try:
        # 各コンポーネントを初期化
        detector = RTDETRv2Detector(conf_threshold=0.5, device="cpu")
        tracker = SimpleTracker()
        
        print("✓ 全コンポーネント初期化完了")
        
        # 最初の5フレームをテスト
        cap = cv2.VideoCapture(video_path)
        
        for frame_idx in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 検出
            detections = detector.detect(frame)
            
            # 馬のみフィルタリング
            horse_detections = [d for d in detections if d['class_name'] == 'horse']
            
            # トラッキング
            tracks = tracker.update(horse_detections)
            
            # 描画
            result_frame = draw_tracks(frame, tracks)
            
            print(f"  フレーム {frame_idx+1}: 検出数={len(detections)}, 馬={len(horse_detections)}, トラック数={len(tracks)}")
        
        cap.release()
        print("✓ 統合テスト完了")
        
        return True
        
    except Exception as e:
        print(f"✗ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト関数"""
    print("RT-DETRv2 MOTシステム テスト開始\n")
    
    tests = [
        ("検出器", test_detector),
        ("トラッカー", test_tracker), 
        ("動画処理", test_video_processor),
        ("統合テスト", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"{test_name} テスト実行中...")
        print('='*50)
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            print(f"✓ {test_name} テスト: 成功")
        else:
            print(f"✗ {test_name} テスト: 失敗")
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("テスト結果サマリー")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<15}: {status}")
        if result:
            passed += 1
    
    print(f"\n合計: {passed}/{len(results)} テストが成功")
    
    if passed == len(results):
        print("\n🎉 すべてのテストが成功しました！")
        print("\n次のコマンドでシステムを実行できます:")
        print("python src/main.py --input videos/realhorses.mp4 --target-class horse --display")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} 個のテストが失敗しました。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
