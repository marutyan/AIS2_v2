"""
RT-DETRv2を使用した物体検出器の実装
軽量モデル（r18vd）を使用してMacBook M2に最適化
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import requests
import os
from pathlib import Path


class RTDETRv2Detector:
    """RT-DETRv2による物体検出器"""
    
    def __init__(self, 
                 model_name: str = "rtdetrv2_r18vd_120e_coco",
                 conf_threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: 使用するモデル名（軽量モデルを使用）
            conf_threshold: 信頼度の閾値
            device: 使用するデバイス（None の場合は自動選択）
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        
        # デバイスの設定（MacBook M2の場合はMPSを優先）
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # COCO クラス名
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # モデルの初期化
        self.model = None
        self.input_size = (640, 640)  # 軽量化のため小さなサイズを使用
        
        # 前処理の定義
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self._load_model()
    
    def _load_model(self):
        """モデルを読み込む"""
        try:
            # RT-DETRv2の軽量モデルを使用
            import torch.hub
            
            # HuggingFace Hub経由でRT-DETRv2モデルを読み込み
            self.model = torch.hub.load(
                'lyuwenyu/RT-DETR', 
                'rtdetrv2_r18vd', 
                pretrained=True,
                source='github'
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"RT-DETRv2 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading RT-DETRv2 model: {e}")
            print("Trying alternative loading method...")
            
            try:
                # 代替方法: Ultralyticsの RTDETR を使用
                from ultralytics import RTDETR
                self.model = RTDETR('rtdetr-l.pt')  # 軽量版
                self.model.to(self.device)
                print(f"Fallback: Ultralytics RTDETR model loaded on {self.device}")
                
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise RuntimeError(f"Failed to load any RT-DETR model. Original error: {e}, Fallback error: {e2}")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        画像を前処理する
        
        Args:
            image: OpenCV形式の画像 (BGR)
            
        Returns:
            前処理済みのテンソル
        """
        # BGRからRGBに変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # 前処理を適用
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0)  # バッチ次元を追加
        
        return tensor.to(self.device)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        物体検出を実行する
        
        Args:
            image: 入力画像 (OpenCV形式, BGR)
            
        Returns:
            検出結果のリスト
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        original_height, original_width = image.shape[:2]
        
        # 前処理
        input_tensor = self._preprocess_image(image)
        
        # 推論実行
        with torch.no_grad():
            try:
                # RT-DETRv2の場合
                outputs = self.model(input_tensor)
                
                # RT-DETRv2の出力形式を処理
                if isinstance(outputs, dict):
                    # RT-DETRv2の標準出力形式
                    pred_boxes = outputs.get('pred_boxes', None)
                    pred_logits = outputs.get('pred_logits', None)
                    
                    if pred_boxes is not None and pred_logits is not None:
                        return self._process_rtdetr_outputs(pred_boxes, pred_logits, original_width, original_height)
                
                # Ultralyticsの場合
                if hasattr(outputs, '__iter__') and hasattr(outputs[0], 'boxes'):
                    return self._process_ultralytics_outputs(outputs)
                
                # 他の形式の場合
                return self._process_generic_outputs(outputs, original_width, original_height)
                
            except Exception as e:
                print(f"Detection error: {e}")
                return []
    
    def _process_rtdetr_outputs(self, pred_boxes, pred_logits, orig_width, orig_height):
        """RT-DETRv2の出力を処理"""
        detections = []
        
        # ソフトマックスでクラス確率を計算
        pred_probs = torch.softmax(pred_logits, dim=-1)
        
        # バッチの最初の結果を取得
        if pred_boxes.dim() == 3:
            pred_boxes = pred_boxes[0]
            pred_probs = pred_probs[0]
        
        for i in range(pred_boxes.shape[0]):
            # 最も確率の高いクラスを取得（背景クラスを除く）
            class_probs = pred_probs[i, :-1]  # 最後のクラスは背景
            max_prob, class_id = torch.max(class_probs, dim=0)
            
            if max_prob.item() >= self.conf_threshold:
                # バウンディングボックス（中心座標 + 幅高さ → 左上右下座標）
                box = pred_boxes[i]
                cx, cy, w, h = box
                
                # 正規化座標を元の画像座標に変換
                x1 = int((cx - w/2) * orig_width)
                y1 = int((cy - h/2) * orig_height)
                x2 = int((cx + w/2) * orig_width)
                y2 = int((cy + h/2) * orig_height)
                
                # 画像境界内にクリップ
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': max_prob.item(),
                    'class_id': class_id.item(),
                    'class_name': self.coco_classes[class_id.item()] if class_id.item() < len(self.coco_classes) else 'unknown'
                }
                detections.append(detection)
        
        return detections
    
    def _process_ultralytics_outputs(self, results):
        """Ultralyticsの出力を処理"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # バウンディングボックス
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    
                    # 信頼度
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # クラスID
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # 信頼度フィルタリング
                    if confidence >= self.conf_threshold:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.coco_classes[class_id] if class_id < len(self.coco_classes) else 'unknown'
                        }
                        detections.append(detection)
        
        return detections
    
    def _process_generic_outputs(self, outputs, orig_width, orig_height):
        """汎用的な出力処理"""
        # 基本的な処理を実装
        detections = []
        
        if torch.is_tensor(outputs):
            # テンソル出力の場合の基本的な処理
            # 形状に基づいて推測
            if outputs.dim() == 3 and outputs.shape[-1] >= 6:  # [batch, num_detections, 6+]
                batch_outputs = outputs[0] if outputs.shape[0] == 1 else outputs
                
                for detection in batch_outputs:
                    if len(detection) >= 6:
                        x1, y1, x2, y2, conf, cls = detection[:6]
                        
                        if conf >= self.conf_threshold:
                            detection_dict = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': self.coco_classes[int(cls)] if int(cls) < len(self.coco_classes) else 'unknown'
                            }
                            detections.append(detection_dict)
        
        return detections
    
    def detect_horses(self, image: np.ndarray) -> List[Dict]:
        """
        馬のみを検出する
        
        Args:
            image: 入力画像 (OpenCV形式, BGR)
            
        Returns:
            馬の検出結果のリスト
        """
        all_detections = self.detect(image)
        horse_detections = [d for d in all_detections if d['class_name'] == 'horse']
        return horse_detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       target_class: Optional[str] = None) -> np.ndarray:
        """
        検出結果を画像に描画する
        
        Args:
            image: 入力画像
            detections: 検出結果
            target_class: 描画する特定のクラス名（Noneの場合は全て描画）
            
        Returns:
            描画済み画像
        """
        result_image = image.copy()
        
        for detection in detections:
            # 特定のクラスのみ描画する場合
            if target_class and detection['class_name'] != target_class:
                continue
                
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # バウンディングボックスを描画
            color = (0, 255, 0) if class_name == 'horse' else (255, 0, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # ラベルを描画
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # ラベル背景
            cv2.rectangle(result_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # ラベルテキスト
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
