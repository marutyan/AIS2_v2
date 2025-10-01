"""
シンプルな物体追跡システム
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist
import cv2


class Track:
    """単一トラックを表すクラス"""
    
    def __init__(self, track_id: int, detection: Dict, max_age: int = 30):
        """
        初期化
        
        Args:
            track_id: トラックID
            detection: 初期検出結果
            max_age: トラックの最大年齢（フレーム数）
        """
        self.track_id = track_id
        self.max_age = max_age
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        
        # 検出履歴
        self.detections = [detection]
        self.positions = [self._get_center(detection['bbox'])]
        
        # 予測用の簡単な速度推定
        self.velocity = np.array([0.0, 0.0])
        
    def _get_center(self, bbox: List[int]) -> np.ndarray:
        """バウンディングボックスの中心点を取得"""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def update(self, detection: Dict):
        """トラックを更新"""
        self.age = 0
        self.hits += 1
        self.hit_streak += 1
        
        # 新しい検出を追加
        self.detections.append(detection)
        new_position = self._get_center(detection['bbox'])
        
        # 速度を更新（簡単な差分）
        if len(self.positions) > 0:
            self.velocity = new_position - self.positions[-1]
        
        self.positions.append(new_position)
        
        # 履歴の長さを制限
        max_history = 10
        if len(self.detections) > max_history:
            self.detections = self.detections[-max_history:]
            self.positions = self.positions[-max_history:]
    
    def predict(self) -> np.ndarray:
        """次の位置を予測"""
        if len(self.positions) == 0:
            return np.array([0, 0])
        
        # 簡単な線形予測
        return self.positions[-1] + self.velocity
    
    def increment_age(self):
        """年齢を増加"""
        self.age += 1
        self.hit_streak = 0
    
    def is_valid(self) -> bool:
        """トラックが有効かどうか"""
        return self.age < self.max_age
    
    def get_current_detection(self) -> Optional[Dict]:
        """現在の検出結果を取得"""
        return self.detections[-1] if self.detections else None


class SimpleTracker:
    """シンプルな物体追跡器"""
    
    def __init__(self, 
                 max_distance: float = 100.0,
                 max_age: int = 30,
                 min_hits: int = 3):
        """
        初期化
        
        Args:
            max_distance: マッチングの最大距離
            max_age: トラックの最大年齢
            min_hits: 有効なトラックとして認識する最小ヒット数
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
    
    def _compute_distances(self, detections: List[Dict]) -> np.ndarray:
        """検出結果とトラック間の距離を計算"""
        if not self.tracks or not detections:
            return np.array([]).reshape(0, 0)
        
        # トラックの予測位置
        track_positions = np.array([track.predict() for track in self.tracks])
        
        # 検出結果の位置
        detection_positions = np.array([
            [(det['bbox'][0] + det['bbox'][2]) / 2, 
             (det['bbox'][1] + det['bbox'][3]) / 2] 
            for det in detections
        ])
        
        # ユークリッド距離を計算
        distances = cdist(track_positions, detection_positions)
        
        return distances
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        ハンガリアン法による割り当て（簡易版）
        
        Returns:
            (matches, unmatched_tracks, unmatched_detections)
        """
        if cost_matrix.size == 0:
            return [], list(range(len(self.tracks))), list(range(len(cost_matrix[0]) if len(cost_matrix) > 0 else 0))
        
        # 簡単な貪欲法による割り当て
        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_detections = list(range(cost_matrix.shape[1]))
        
        # 距離が閾値以下のペアを見つける
        valid_pairs = np.where(cost_matrix < self.max_distance)
        
        # 距離順にソート
        if len(valid_pairs[0]) > 0:
            distances = cost_matrix[valid_pairs]
            sorted_indices = np.argsort(distances)
            
            for idx in sorted_indices:
                track_idx = valid_pairs[0][idx]
                det_idx = valid_pairs[1][idx]
                
                if track_idx in unmatched_tracks and det_idx in unmatched_detections:
                    matches.append((track_idx, det_idx))
                    unmatched_tracks.remove(track_idx)
                    unmatched_detections.remove(det_idx)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        トラッカーを更新
        
        Args:
            detections: 新しい検出結果のリスト
            
        Returns:
            更新されたトラックのリスト
        """
        self.frame_count += 1
        
        # 距離行列を計算
        cost_matrix = self._compute_distances(detections)
        
        # 割り当てを実行
        matches, unmatched_tracks, unmatched_detections = self._hungarian_assignment(cost_matrix)
        
        # マッチしたトラックを更新
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])
        
        # マッチしなかったトラックの年齢を増加
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].increment_age()
        
        # 新しいトラックを作成
        for det_idx in unmatched_detections:
            new_track = Track(self.next_track_id, detections[det_idx], self.max_age)
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # 古いトラックを削除
        self.tracks = [track for track in self.tracks if track.is_valid()]
        
        # 有効なトラックのみを返す
        valid_tracks = [track for track in self.tracks 
                       if track.hits >= self.min_hits or track.hit_streak >= 1]
        
        return valid_tracks
    
    def get_active_tracks(self) -> List[Track]:
        """アクティブなトラックを取得"""
        return [track for track in self.tracks 
                if track.hits >= self.min_hits and track.age < 5]


def draw_tracks(image: np.ndarray, tracks: List[Track], 
                draw_trail: bool = True, trail_length: int = 10) -> np.ndarray:
    """
    トラックを画像に描画
    
    Args:
        image: 入力画像
        tracks: トラックのリスト
        draw_trail: 軌跡を描画するかどうか
        trail_length: 軌跡の長さ
        
    Returns:
        描画済み画像
    """
    result_image = image.copy()
    
    # カラーパレット
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 0, 0)
    ]
    
    for track in tracks:
        current_detection = track.get_current_detection()
        if not current_detection:
            continue
        
        color = colors[track.track_id % len(colors)]
        
        # バウンディングボックスを描画
        x1, y1, x2, y2 = current_detection['bbox']
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # トラックIDとクラス名を描画
        label = f"ID:{track.track_id} {current_detection['class_name']}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # ラベル背景
        cv2.rectangle(result_image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # ラベルテキスト
        cv2.putText(result_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 軌跡を描画
        if draw_trail and len(track.positions) > 1:
            points = track.positions[-trail_length:]
            for i in range(1, len(points)):
                pt1 = tuple(map(int, points[i-1]))
                pt2 = tuple(map(int, points[i]))
                thickness = max(1, int(3 * (i / len(points))))
                cv2.line(result_image, pt1, pt2, color, thickness)
    
    return result_image
