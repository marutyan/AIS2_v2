"""
動画処理ユーティリティ
"""

import cv2
import numpy as np
from typing import Optional, Callable, Generator, Tuple
from pathlib import Path
import time


class VideoProcessor:
    """動画処理クラス"""
    
    def __init__(self, input_path: str, output_path: Optional[str] = None):
        """
        初期化
        
        Args:
            input_path: 入力動画のパス
            output_path: 出力動画のパス（Noneの場合は保存しない）
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else None
        
        # 動画キャプチャを初期化
        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")
        
        # 動画の基本情報を取得
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"Video info: {self.width}x{self.height}, {self.fps}FPS, {self.total_frames} frames, {self.duration:.2f}s")
        
        # 動画ライターを初期化
        self.writer = None
        if self.output_path:
            self._init_writer()
    
    def _init_writer(self):
        """動画ライターを初期化"""
        if self.output_path:
            # 出力ディレクトリを作成
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # コーデックを設定（MacOS対応）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                raise ValueError(f"Cannot create video writer: {self.output_path}")
    
    def process_frames(self, 
                      processor_func: Callable[[np.ndarray, int], np.ndarray],
                      show_progress: bool = True) -> None:
        """
        フレームごとに処理を実行
        
        Args:
            processor_func: フレーム処理関数 (frame, frame_idx) -> processed_frame
            show_progress: 進捗表示の有無
        """
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # フレーム処理
                processed_frame = processor_func(frame, frame_idx)
                
                # 結果を保存
                if self.writer:
                    self.writer.write(processed_frame)
                
                # 進捗表示
                if show_progress and frame_idx % 30 == 0:  # 30フレームごとに表示
                    elapsed = time.time() - start_time
                    progress = (frame_idx + 1) / self.total_frames * 100
                    fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                    print(f"Progress: {progress:.1f}% ({frame_idx+1}/{self.total_frames}), FPS: {fps:.1f}")
                
                frame_idx += 1
        
        finally:
            self.release()
    
    def process_frames_generator(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        フレームを順次生成するジェネレーター
        
        Yields:
            (frame, frame_index) のタプル
        """
        frame_idx = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                yield frame, frame_idx
                frame_idx += 1
        
        finally:
            self.cap.release()
    
    def save_frame(self, frame: np.ndarray):
        """フレームを保存"""
        if self.writer:
            self.writer.write(frame)
    
    def release(self):
        """リソースを解放"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoDisplay:
    """動画表示ユーティリティ"""
    
    @staticmethod
    def show_frame(frame: np.ndarray, window_name: str = "Video", wait_key: int = 1) -> bool:
        """
        フレームを表示
        
        Args:
            frame: 表示するフレーム
            window_name: ウィンドウ名
            wait_key: キー待機時間（ミリ秒）
            
        Returns:
            継続するかどうか（'q'キーが押された場合はFalse）
        """
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(wait_key) & 0xFF
        return key != ord('q')
    
    @staticmethod
    def resize_for_display(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
        """
        表示用にフレームをリサイズ
        
        Args:
            frame: 入力フレーム
            max_width: 最大幅
            max_height: 最大高さ
            
        Returns:
            リサイズされたフレーム
        """
        height, width = frame.shape[:2]
        
        # アスペクト比を維持してリサイズ
        scale = min(max_width / width, max_height / height)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        
        return frame


def create_output_path(input_path: str, suffix: str = "_processed") -> str:
    """
    出力パスを作成
    
    Args:
        input_path: 入力ファイルのパス
        suffix: ファイル名に追加するサフィックス
        
    Returns:
        出力ファイルのパス
    """
    input_path = Path(input_path)
    output_name = input_path.stem + suffix + input_path.suffix
    return str(input_path.parent / output_name)
