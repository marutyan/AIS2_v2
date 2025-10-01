#!/usr/bin/env python3
"""
RT-DETRv2を使用したMOT（Multi-Object Tracking）システム
メインエントリポイント
"""

import argparse
import sys
from pathlib import Path
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mot_system.detection.rt_detr_detector import RTDETRv2Detector
from src.mot_system.tracking.simple_tracker import SimpleTracker, draw_tracks
from src.mot_system.utils.video_processor import VideoProcessor, create_output_path, VideoDisplay


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="RT-DETRv2 MOT System")
    parser.add_argument("--input", "-i", required=True, help="入力動画のパス")
    parser.add_argument("--output", "-o", help="出力動画のパス（指定しない場合は自動生成）")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="信頼度の閾値")
    parser.add_argument("--target-class", "-t", default="horse", help="追跡対象のクラス名")
    parser.add_argument("--display", "-d", action="store_true", help="リアルタイム表示")
    parser.add_argument("--save-frames", "-s", action="store_true", help="フレームを保存")
    parser.add_argument("--device", help="使用するデバイス（cpu, cuda, mps）")
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 入力ファイルが見つかりません: {args.input}")
        return 1
    
    # 出力パスの設定
    if args.output:
        output_path = args.output
    else:
        output_path = create_output_path(args.input, "_mot_result")
    
    print(f"入力: {args.input}")
    print(f"出力: {output_path}")
    print(f"対象クラス: {args.target_class}")
    print(f"信頼度閾値: {args.conf}")
    
    try:
        # 検出器を初期化
        print("RT-DETRv2検出器を初期化中...")
        detector = RTDETRv2Detector(
            conf_threshold=args.conf,
            device=args.device
        )
        
        # トラッカーを初期化
        print("トラッカーを初期化中...")
        tracker = SimpleTracker(
            max_distance=100.0,
            max_age=30,
            min_hits=3
        )
        
        # 動画処理を初期化
        print("動画処理を開始...")
        save_video = args.save_frames or not args.display
        
        with VideoProcessor(
            str(input_path), 
            output_path if save_video else None
        ) as video_processor:
            
            def process_frame(frame, frame_idx):
                """フレーム処理関数"""
                start_time = time.time()
                
                # 物体検出
                detections = detector.detect(frame)
                
                # 特定のクラスのみフィルタリング
                if args.target_class:
                    target_detections = [
                        d for d in detections 
                        if d['class_name'] == args.target_class
                    ]
                else:
                    target_detections = detections
                
                # トラッキング
                tracks = tracker.update(target_detections)
                
                # 結果を描画
                result_frame = draw_tracks(
                    frame, 
                    tracks, 
                    draw_trail=True, 
                    trail_length=15
                )
                
                # 処理時間を表示
                processing_time = time.time() - start_time
                fps_text = f"FPS: {1.0/processing_time:.1f}" if processing_time > 0 else "FPS: N/A"
                
                # 統計情報を描画
                stats_text = [
                    fps_text,
                    f"Frame: {frame_idx + 1}",
                    f"Detections: {len(target_detections)}",
                    f"Tracks: {len(tracks)}"
                ]
                
                y_offset = 30
                for text in stats_text:
                    cv2.putText(result_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # リアルタイム表示
                if args.display:
                    display_frame = VideoDisplay.resize_for_display(result_frame)
                    if not VideoDisplay.show_frame(display_frame, "MOT Result", 1):
                        return None  # 'q'キーが押された場合は終了
                
                return result_frame
            
            # フレーム処理を実行
            video_processor.process_frames(process_frame, show_progress=True)
        
        print(f"\n処理完了！")
        if save_video:
            print(f"結果を保存しました: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        return 1
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import cv2
    sys.exit(main())
