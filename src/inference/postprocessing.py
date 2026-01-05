# src/inference/postprocessing.py
"""
Detection Postprocessing for Robotic Vision

Features:
1. Non-Maximum Suppression (NMS) with manifold constraints
2. Detection filtering and validation
3. Object tracking across frames
4. Detection fusion from multiple scales
5. Confidence calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque, defaultdict
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class PostprocessingConfig:
    """Configuration for postprocessing."""
    # NMS parameters
    nms_iou_threshold: float = 0.45
    nms_score_threshold: float = 0.25
    nms_max_detections: int = 100
    nms_method: str = "standard"  # standard, soft, matrix
    
    # Detection filtering
    min_confidence: float = 0.1
    min_box_size: int = 4
    max_box_size: int = 1000
    aspect_ratio_range: Tuple[float, float] = (0.1, 10.0)
    
    # Object tracking
    enable_tracking: bool = True
    tracker_type: str = "sort"  # sort, deepsort, iou
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    
    # Multi-scale fusion
    enable_multi_scale_fusion: bool = True
    scale_weights: List[float] = None  # Weights for different scales
    
    # Confidence calibration
    enable_calibration: bool = True
    calibration_temperature: float = 1.0
    
    # Robot-specific constraints
    max_detections_per_frame: int = 50
    detection_buffer_size: int = 10
    
    def __post_init__(self):
        if self.scale_weights is None:
            self.scale_weights = [0.4, 0.3, 0.3]  # small, medium, large

class NMSMethod(Enum):
    """Non-Maximum Suppression methods."""
    STANDARD = "standard"
    SOFT = "soft"
    MATRIX = "matrix"
    CLUSTER = "cluster"

class DetectionPostprocessor:
    """
    Advanced detection postprocessor for robotic vision.
    
    Features:
    1. Multi-scale detection fusion
    2. Advanced NMS with different methods
    3. Detection filtering and validation
    4. Confidence calibration
    5. Output formatting for robot control
    """
    
    def __init__(self, config: Optional[PostprocessingConfig] = None):
        """
        Initialize detection postprocessor.
        
        Args:
            config: Postprocessing configuration
        """
        self.config = config or PostprocessingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.nms_filter = NMSFilter(self.config)
        if self.config.enable_tracking:
            self.tracker = DetectionTracker(self.config)
        
        # Detection history for temporal filtering
        self.detection_history = deque(maxlen=self.config.detection_buffer_size)
        self.frame_counter = 0
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        
        logger.info("DetectionPostprocessor initialized")
        logger.info(f"NMS threshold: {self.config.nms_iou_threshold}")
        logger.info(f"Tracking enabled: {self.config.enable_tracking}")
    
    def process(self, model_outputs: Dict[str, Any], 
                image_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Process model outputs into final detections.
        
        Args:
            model_outputs: Raw model outputs
            image_size: Original image size (height, width)
            
        Returns:
            Processed detections with metadata
        """
        start_time = time.perf_counter()
        
        # Extract predictions from model outputs
        predictions = self._extract_predictions(model_outputs)
        
        if not predictions:
            return self._create_empty_result()
        
        # Apply multi-scale fusion if enabled
        if self.config.enable_multi_scale_fusion:
            predictions = self._fuse_multi_scale_predictions(predictions)
        
        # Convert to unified format
        boxes, scores, class_ids = self._convert_to_unified_format(predictions)
        
        if len(boxes) == 0:
            return self._create_empty_result()
        
        # Apply confidence calibration
        if self.config.enable_calibration:
            scores = self._calibrate_confidences(scores)
        
        # Filter by confidence threshold
        mask = scores >= self.config.min_confidence
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return self._create_empty_result()
        
        # Apply NMS
        keep_indices = self.nms_filter.apply(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids
        )
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
        
        # Filter invalid boxes
        boxes, scores, class_ids = self._filter_invalid_boxes(
            boxes, scores, class_ids
        )
        
        # Limit number of detections
        if len(boxes) > self.config.max_detections_per_frame:
            # Keep top-K by confidence
            top_k = self.config.max_detections_per_frame
            indices = torch.argsort(scores, descending=True)[:top_k]
            boxes = boxes[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
        
        # Scale boxes to original image size if provided
        if image_size is not None:
            boxes = self._scale_boxes_to_image(boxes, image_size)
        
        # Apply tracking if enabled
        if self.config.enable_tracking:
            tracked_detections = self.tracker.update(
                boxes=boxes.cpu().numpy(),
                scores=scores.cpu().numpy(),
                class_ids=class_ids.cpu().numpy(),
                frame_id=self.frame_counter
            )
        else:
            tracked_detections = {
                'boxes': boxes.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'class_ids': class_ids.cpu().numpy(),
                'track_ids': np.arange(len(boxes)),
            }
        
        # Update frame counter
        self.frame_counter += 1
        
        # Update detection history
        self._update_detection_history(tracked_detections)
        
        # Calculate processing time
        processing_time = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        # Prepare final result
        result = self._format_result(tracked_detections, processing_time)
        
        return result
    
    def _extract_predictions(self, model_outputs: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Extract predictions from model outputs."""
        predictions = []
        
        # Handle different output formats
        if 'detections' in model_outputs:
            # Direct detection output
            det_output = model_outputs['detections']
            
            if isinstance(det_output, torch.Tensor):
                # Single scale output
                predictions.append({
                    'boxes': det_output[..., :4],
                    'scores': det_output[..., 4],
                    'class_probs': det_output[..., 5:],
                    'scale_weight': 1.0
                })
            elif isinstance(det_output, dict):
                # Multi-scale output
                for scale_name, scale_output in det_output.items():
                    if isinstance(scale_output, torch.Tensor):
                        weight = self._get_scale_weight(scale_name)
                        predictions.append({
                            'boxes': scale_output[..., :4],
                            'scores': scale_output[..., 4],
                            'class_probs': scale_output[..., 5:],
                            'scale_weight': weight
                        })
        
        # Handle YOLO-style outputs
        elif 'yolo_detections' in model_outputs:
            yolo_outputs = model_outputs['yolo_detections']
            
            for scale_name, scale_output in yolo_outputs.items():
                if isinstance(scale_output, torch.Tensor):
                    weight = self._get_scale_weight(scale_name)
                    
                    # Reshape if needed
                    if len(scale_output.shape) == 5:  # [B, A, H, W, C]
                        batch_size, num_anchors, height, width, _ = scale_output.shape
                        scale_output = scale_output.view(
                            batch_size, num_anchors * height * width, -1
                        )
                    
                    predictions.append({
                        'boxes': scale_output[..., :4],
                        'scores': scale_output[..., 4:5],
                        'class_probs': scale_output[..., 5:],
                        'scale_weight': weight
                    })
        
        return predictions
    
    def _get_scale_weight(self, scale_name: str) -> float:
        """Get weight for scale based on name."""
        scale_name = scale_name.lower()
        
        if 'small' in scale_name:
            return self.config.scale_weights[0]
        elif 'medium' in scale_name:
            return self.config.scale_weights[1]
        elif 'large' in scale_name:
            return self.config.scale_weights[2]
        else:
            return 1.0
    
    def _fuse_multi_scale_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Fuse predictions from multiple scales."""
        if len(predictions) <= 1:
            return predictions
        
        # Weight predictions by scale
        fused_predictions = []
        
        for pred in predictions:
            weight = pred.get('scale_weight', 1.0)
            
            # Adjust scores and class probabilities
            if 'scores' in pred and pred['scores'] is not None:
                pred['scores'] = pred['scores'] * weight
            
            if 'class_probs' in pred and pred['class_probs'] is not None:
                pred['class_probs'] = pred['class_probs'] * weight
            
            fused_predictions.append(pred)
        
        return fused_predictions
    
    def _convert_to_unified_format(self, predictions: List[Dict]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert predictions to unified format for processing."""
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        for pred in predictions:
            boxes = pred.get('boxes')
            scores = pred.get('scores')
            class_probs = pred.get('class_probs')
            
            if boxes is None or scores is None or class_probs is None:
                continue
            
            # Ensure correct shapes
            if len(boxes.shape) == 3:  # [B, N, 4]
                boxes = boxes.view(-1, 4)
                scores = scores.view(-1)
                class_probs = class_probs.view(-1, class_probs.shape[-1])
            
            # Get class IDs from probabilities
            if class_probs.shape[-1] > 1:  # Multi-class
                class_scores, class_ids = torch.max(class_probs, dim=-1)
                # Combine objectness and class scores
                combined_scores = scores * class_scores
            else:  # Single class or objectness only
                class_ids = torch.zeros_like(scores, dtype=torch.long)
                combined_scores = scores
            
            # Filter by score threshold
            mask = combined_scores >= self.config.nms_score_threshold
            if mask.any():
                all_boxes.append(boxes[mask])
                all_scores.append(combined_scores[mask])
                all_class_ids.append(class_ids[mask])
        
        if not all_boxes:
            return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
        
        # Concatenate all predictions
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        class_ids = torch.cat(all_class_ids, dim=0)
        
        return boxes, scores, class_ids
    
    def _calibrate_confidences(self, scores: torch.Tensor) -> torch.Tensor:
        """Calibrate confidence scores using temperature scaling."""
        # Apply temperature scaling
        if self.config.calibration_temperature != 1.0:
            scores = torch.sigmoid(
                torch.logit(scores) / self.config.calibration_temperature
            )
        
        return scores
    
    def _filter_invalid_boxes(self, boxes: torch.Tensor, scores: torch.Tensor,
                             class_ids: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter invalid bounding boxes."""
        if len(boxes) == 0:
            return boxes, scores, class_ids
        
        # Extract box parameters
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]
        
        # Convert to corner coordinates for filtering
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Create validity masks
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
        
        # Check box dimensions
        valid_mask &= (width >= self.config.min_box_size)
        valid_mask &= (height >= self.config.min_box_size)
        valid_mask &= (width <= self.config.max_box_size)
        valid_mask &= (height <= self.config.max_box_size)
        
        # Check aspect ratio
        aspect_ratio = width / (height + 1e-6)
        valid_mask &= (aspect_ratio >= self.config.aspect_ratio_range[0])
        valid_mask &= (aspect_ratio <= self.config.aspect_ratio_range[1])
        
        # Check coordinates
        valid_mask &= (x1 >= 0) & (x2 <= 1)  # Normalized coordinates
        valid_mask &= (y1 >= 0) & (y2 <= 1)
        
        # Check area
        area = width * height
        valid_mask &= (area > 0)
        
        # Apply mask
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        return boxes, scores, class_ids
    
    def _scale_boxes_to_image(self, boxes: torch.Tensor, 
                             image_size: Tuple[int, int]) -> torch.Tensor:
        """Scale normalized boxes to image coordinates."""
        img_h, img_w = image_size
        
        # Boxes are in normalized format [x_center, y_center, width, height]
        scaled_boxes = boxes.clone()
        
        # Scale center coordinates
        scaled_boxes[:, 0] = scaled_boxes[:, 0] * img_w
        scaled_boxes[:, 1] = scaled_boxes[:, 1] * img_h
        
        # Scale dimensions
        scaled_boxes[:, 2] = scaled_boxes[:, 2] * img_w
        scaled_boxes[:, 3] = scaled_boxes[:, 3] * img_h
        
        return scaled_boxes
    
    def _update_detection_history(self, detections: Dict[str, np.ndarray]):
        """Update detection history for temporal analysis."""
        history_entry = {
            'frame_id': self.frame_counter,
            'timestamp': time.time(),
            'detections': detections.copy()
        }
        self.detection_history.append(history_entry)
    
    def _format_result(self, detections: Dict[str, np.ndarray], 
                      processing_time: float) -> Dict[str, Any]:
        """Format detection results."""
        result = {
            'detections': {
                'boxes': detections['boxes'].tolist(),
                'scores': detections['scores'].tolist(),
                'class_ids': detections['class_ids'].tolist(),
                'track_ids': detections['track_ids'].tolist(),
            },
            'metadata': {
                'frame_id': self.frame_counter,
                'num_detections': len(detections['boxes']),
                'processing_time_ms': processing_time,
                'timestamp': time.time(),
            },
            'performance': {
                'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
                'fps': 1000 / np.mean(self.processing_times) if self.processing_times else 0,
            }
        }
        
        # Add tracking info if available
        if hasattr(self, 'tracker'):
            tracking_info = self.tracker.get_tracking_info()
            result['tracking'] = tracking_info
        
        return result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no detections."""
        return {
            'detections': {
                'boxes': [],
                'scores': [],
                'class_ids': [],
                'track_ids': [],
            },
            'metadata': {
                'frame_id': self.frame_counter,
                'num_detections': 0,
                'processing_time_ms': 0,
                'timestamp': time.time(),
            },
            'performance': {
                'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
                'fps': 1000 / np.mean(self.processing_times) if self.processing_times else 0,
            }
        }
    
    def reset(self):
        """Reset postprocessor state."""
        self.detection_history.clear()
        self.frame_counter = 0
        self.processing_times.clear()
        
        if hasattr(self, 'tracker'):
            self.tracker.reset()
        
        logger.info("DetectionPostprocessor reset")

class NMSFilter:
    """Non-Maximum Suppression implementation with multiple methods."""
    
    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def apply(self, boxes: torch.Tensor, scores: torch.Tensor,
              class_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply NMS to detections.
        
        Args:
            boxes: Bounding boxes [N, 4] in format (x_center, y_center, width, height)
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Convert to corner format for NMS
        boxes_corner = self._center_to_corner(boxes)
        
        # Select NMS method
        nms_method = NMSMethod(self.config.nms_method)
        
        if nms_method == NMSMethod.STANDARD:
            keep_indices = self._standard_nms(boxes_corner, scores, class_ids)
        elif nms_method == NMSMethod.SOFT:
            keep_indices = self._soft_nms(boxes_corner, scores, class_ids)
        elif nms_method == NMSMethod.MATRIX:
            keep_indices = self._matrix_nms(boxes_corner, scores, class_ids)
        elif nms_method == NMSMethod.CLUSTER:
            keep_indices = self._cluster_nms(boxes_corner, scores, class_ids)
        else:
            raise ValueError(f"Unknown NMS method: {nms_method}")
        
        return keep_indices
    
    def _center_to_corner(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format to corner format."""
        x_center, y_center, width, height = boxes.unbind(-1)
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _corner_to_center(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert corner format to center format."""
        x1, y1, x2, y2 = boxes.unbind(-1)
        
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        return torch.stack([x_center, y_center, width, height], dim=-1)
    
    def _standard_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                     class_ids: torch.Tensor) -> torch.Tensor:
        """Standard NMS implementation."""
        # Sort by score
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_boxes = boxes[sorted_indices]
        sorted_class_ids = class_ids[sorted_indices]
        
        keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        
        for i in range(len(sorted_boxes)):
            if not keep_mask[i]:
                continue
            
            # Compute IoU with remaining boxes
            box_i = sorted_boxes[i].unsqueeze(0)
            remaining_mask = keep_mask & (torch.arange(len(sorted_boxes), device=boxes.device) > i)
            
            if not remaining_mask.any():
                break
            
            remaining_boxes = sorted_boxes[remaining_mask]
            remaining_class_ids = sorted_class_ids[remaining_mask]
            
            # Only suppress boxes of the same class
            same_class_mask = (remaining_class_ids == sorted_class_ids[i])
            
            if same_class_mask.any():
                # Compute IoU
                ious = self._compute_iou(box_i, remaining_boxes[same_class_mask])
                
                # Suppress boxes with high IoU
                suppress_mask = ious > self.config.nms_iou_threshold
                
                # Map back to original indices
                suppress_indices = torch.where(remaining_mask)[0][torch.where(same_class_mask)[0][suppress_mask]]
                keep_mask[suppress_indices] = False
        
        # Convert to indices
        keep_indices = sorted_indices[keep_mask]
        
        # Limit number of detections
        if len(keep_indices) > self.config.nms_max_detections:
            keep_indices = keep_indices[:self.config.nms_max_detections]
        
        return keep_indices
    
    def _soft_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                 class_ids: torch.Tensor) -> torch.Tensor:
        """Soft NMS implementation."""
        # Convert to CPU for soft NMS (usually implemented in numpy)
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        class_ids_np = class_ids.cpu().numpy()
        
        # Implementation of soft NMS
        n = len(boxes_np)
        keep_indices = []
        
        for i in range(n):
            max_pos = i + np.argmax(scores_np[i:])
            
            # Swap
            boxes_np[[i, max_pos]] = boxes_np[[max_pos, i]]
            scores_np[[i, max_pos]] = scores_np[[max_pos, i]]
            class_ids_np[[i, max_pos]] = class_ids_np[[max_pos, i]]
            
            # Soft suppression
            for j in range(i + 1, n):
                if class_ids_np[i] != class_ids_np[j]:
                    continue
                
                iou = self._compute_iou_np(boxes_np[i], boxes_np[j])
                
                if iou > self.config.nms_iou_threshold:
                    # Gaussian penalty
                    sigma = 0.5
                    scores_np[j] *= np.exp(-(iou * iou) / sigma)
            
            # Keep if score is above threshold
            if scores_np[i] >= self.config.min_confidence:
                keep_indices.append(i)
        
        # Convert back to tensor
        keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
        
        # Limit number of detections
        if len(keep_indices) > self.config.nms_max_detections:
            keep_indices = keep_indices[:self.config.nms_max_detections]
        
        return keep_indices
    
    def _matrix_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                   class_ids: torch.Tensor) -> torch.Tensor:
        """Matrix NMS implementation."""
        # This is a simplified version
        # In production, you might want a more complete implementation
        
        # Group by class
        unique_classes = torch.unique(class_ids)
        keep_indices = []
        
        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]
            
            if len(cls_boxes) == 0:
                continue
            
            # Compute IoU matrix
            iou_matrix = self._compute_iou_matrix(cls_boxes)
            
            # Initialize decay factors
            decay = torch.ones_like(cls_scores)
            
            # Matrix NMS
            for i in range(len(cls_boxes)):
                # Find boxes with higher scores
                higher_mask = cls_scores > cls_scores[i]
                
                if higher_mask.any():
                    # Get maximum IoU with higher scoring boxes
                    max_iou = torch.max(iou_matrix[i, higher_mask])
                    
                    if max_iou > self.config.nms_iou_threshold:
                        # Apply decay
                        decay[i] *= (1 - max_iou)
            
            # Apply decay to scores
            decayed_scores = cls_scores * decay
            
            # Keep boxes with decayed scores above threshold
            keep_mask = decayed_scores >= self.config.min_confidence
            keep_indices.append(cls_indices[keep_mask])
        
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
        else:
            keep_indices = torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Sort by score
        if len(keep_indices) > 0:
            keep_scores = scores[keep_indices]
            sorted_indices = torch.argsort(keep_scores, descending=True)
            keep_indices = keep_indices[sorted_indices]
        
        # Limit number of detections
        if len(keep_indices) > self.config.nms_max_detections:
            keep_indices = keep_indices[:self.config.nms_max_detections]
        
        return keep_indices
    
    def _cluster_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                    class_ids: torch.Tensor) -> torch.Tensor:
        """Cluster NMS implementation."""
        # Group by class
        unique_classes = torch.unique(class_ids)
        keep_indices = []
        
        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]
            
            if len(cls_boxes) == 0:
                continue
            
            # Sort by score
            sorted_scores, sorted_order = torch.sort(cls_scores, descending=True)
            sorted_boxes = cls_boxes[sorted_order]
            sorted_indices = cls_indices[sorted_order]
            
            # Cluster NMS
            n = len(sorted_boxes)
            keep_mask = torch.ones(n, dtype=torch.bool, device=boxes.device)
            
            for i in range(n):
                if not keep_mask[i]:
                    continue
                
                # Compute IoU with remaining boxes
                if i + 1 < n:
                    ious = self._compute_iou(
                        sorted_boxes[i].unsqueeze(0),
                        sorted_boxes[i+1:][keep_mask[i+1:]]
                    )
                    
                    # Suppress boxes in the same cluster
                    suppress_mask = ious > self.config.nms_iou_threshold
                    
                    # Map back to indices
                    suppress_indices = torch.where(keep_mask[i+1:])[0][suppress_mask]
                    keep_mask[i+1:][suppress_indices] = False
            
            keep_indices.append(sorted_indices[keep_mask])
        
        if keep_indices:
            keep_indices = torch.cat(keep_indices)
        else:
            keep_indices = torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Limit number of detections
        if len(keep_indices) > self.config.nms_max_detections:
            keep_indices = keep_indices[:self.config.nms_max_detections]
        
        return keep_indices
    
    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        # box1: [N, 4], box2: [M, 4]
        # Returns: [N, M]
        
        # Get coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unsqueeze(1).unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unsqueeze(0).unbind(-1)
        
        # Intersection coordinates
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        # Intersection area
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height
        
        # Box areas
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        # Union area
        union_area = b1_area + b2_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    def _compute_iou_matrix(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU matrix for a set of boxes."""
        n = len(boxes)
        iou_matrix = torch.zeros((n, n), device=boxes.device)
        
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._compute_iou_np(
                    boxes[i].cpu().numpy(),
                    boxes[j].cpu().numpy()
                )
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        return iou_matrix
    
    def _compute_iou_np(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes (numpy version)."""
        # Convert to corner format if needed
        if len(box1) == 4 and len(box2) == 4:
            # Assume already in corner format
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
        else:
            raise ValueError("Boxes must be in corner format")
        
        # Intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)

class DetectionTracker:
    """Object tracker for maintaining detection consistency across frames."""
    
    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.trackers = {}
        self.next_id = 0
        self.frame_history = deque(maxlen=100)
        
        # Choose tracker implementation
        tracker_type = config.tracker_type.lower()
        
        if tracker_type == "sort":
            self._init_sort_tracker()
        elif tracker_type == "deepsort":
            self._init_deepsort_tracker()
        elif tracker_type == "iou":
            self._init_iou_tracker()
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
    
    def _init_sort_tracker(self):
        """Initialize SORT tracker."""
        try:
            from sort import Sort
            self.tracker = Sort(
                max_age=self.config.max_age,
                min_hits=self.config.min_hits,
                iou_threshold=self.config.iou_threshold
            )
            self.use_external = True
        except ImportError:
            logger.warning("SORT not available, using simple IoU tracker")
            self._init_iou_tracker()
    
    def _init_deepsort_tracker(self):
        """Initialize DeepSORT tracker."""
        try:
            from deep_sort import DeepSort
            self.tracker = DeepSort(
                max_age=self.config.max_age,
                nn_budget=100,
                max_iou_distance=self.config.iou_threshold
            )
            self.use_external = True
        except ImportError:
            logger.warning("DeepSORT not available, using simple IoU tracker")
            self._init_iou_tracker()
    
    def _init_iou_tracker(self):
        """Initialize simple IoU-based tracker."""
        self.trackers = {}
        self.next_id = 0
        self.use_external = False
    
    def update(self, boxes: np.ndarray, scores: np.ndarray,
               class_ids: np.ndarray, frame_id: int) -> Dict[str, np.ndarray]:
        """Update tracker with new detections."""
        if self.use_external:
            return self._update_external(boxes, scores, class_ids, frame_id)
        else:
            return self._update_iou(boxes, scores, class_ids, frame_id)
    
    def _update_external(self, boxes: np.ndarray, scores: np.ndarray,
                        class_ids: np.ndarray, frame_id: int) -> Dict[str, np.ndarray]:
        """Update using external tracker."""
        # Prepare detections for external tracker
        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            # Convert to [x1, y1, x2, y2, score, class] format
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score, cls_id])
        
        if detections:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 6))
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Parse results
        track_boxes = []
        track_scores = []
        track_class_ids = []
        track_ids = []
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, class_id = obj[:6]
            score = obj[4] if len(obj) > 4 else 1.0
            
            # Convert to center format
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2
            
            track_boxes.append([x_center, y_center, width, height])
            track_scores.append(score)
            track_class_ids.append(class_id)
            track_ids.append(track_id)
        
        return {
            'boxes': np.array(track_boxes) if track_boxes else np.empty((0, 4)),
            'scores': np.array(track_scores) if track_scores else np.empty((0,)),
            'class_ids': np.array(track_class_ids) if track_class_ids else np.empty((0,), dtype=int),
            'track_ids': np.array(track_ids) if track_ids else np.empty((0,), dtype=int),
        }
    
    def _update_iou(self, boxes: np.ndarray, scores: np.ndarray,
                   class_ids: np.ndarray, frame_id: int) -> Dict[str, np.ndarray]:
        """Update using simple IoU-based tracker."""
        # Convert to corner format for IoU calculation
        boxes_corner = self._center_to_corner_np(boxes)
        
        # Match existing tracks
        matched_indices = []
        unmatched_detections = list(range(len(boxes)))
        
        for track_id, track in self.trackers.items():
            if track['class_id'] not in class_ids:
                continue
            
            # Get last known position
            last_box = track['history'][-1]
            last_box_corner = self._center_to_corner_np(last_box.reshape(1, 4))[0]
            
            # Compute IoU with all detections of the same class
            same_class_mask = class_ids == track['class_id']
            if not same_class_mask.any():
                continue
            
            same_class_indices = np.where(same_class_mask)[0]
            same_class_boxes = boxes_corner[same_class_mask]
            
            # Compute IoU
            ious = []
            for i, box in enumerate(same_class_boxes):
                iou = self._compute_iou_np(last_box_corner, box)
                ious.append(iou)
            
            ious = np.array(ious)
            
            # Find best match
            if len(ious) > 0:
                best_idx = np.argmax(ious)
                best_iou = ious[best_idx]
                
                if best_iou > self.config.iou_threshold:
                    detection_idx = same_class_indices[best_idx]
                    
                    if detection_idx in unmatched_detections:
                        matched_indices.append((track_id, detection_idx))
                        unmatched_detections.remove(detection_idx)
        
        # Update matched tracks
        for track_id, detection_idx in matched_indices:
            track = self.trackers[track_id]
            track['history'].append(boxes[detection_idx])
            track['score'] = scores[detection_idx]  # Update with latest score
            track['last_seen'] = frame_id
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.trackers.items():
            if frame_id - track['last_seen'] > self.config.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.trackers[track_id]
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            
            self.trackers[track_id] = {
                'history': [boxes[detection_idx]],
                'score': scores[detection_idx],
                'class_id': class_ids[detection_idx],
                'first_seen': frame_id,
                'last_seen': frame_id,
            }
        
        # Prepare output
        track_boxes = []
        track_scores = []
        track_class_ids = []
        track_ids = []
        
        for track_id, track in self.trackers.items():
            # Use average of last few positions
            history = track['history'][-3:]  # Last 3 positions
            avg_box = np.mean(history, axis=0)
            
            track_boxes.append(avg_box)
            track_scores.append(track['score'])
            track_class_ids.append(track['class_id'])
            track_ids.append(track_id)
        
        return {
            'boxes': np.array(track_boxes) if track_boxes else np.empty((0, 4)),
            'scores': np.array(track_scores) if track_scores else np.empty((0,)),
            'class_ids': np.array(track_class_ids) if track_class_ids else np.empty((0,), dtype=int),
            'track_ids': np.array(track_ids) if track_ids else np.empty((0,), dtype=int),
        }
    
    def _center_to_corner_np(self, boxes: np.ndarray) -> np.ndarray:
        """Convert center format to corner format (numpy)."""
        if len(boxes.shape) == 1:
            boxes = boxes.reshape(1, -1)
        
        x_center, y_center, width, height = boxes.T
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def _compute_iou_np(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes (numpy)."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def get_tracking_info(self) -> Dict[str, Any]:
        """Get tracking information."""
        info = {
            'num_active_tracks': len(self.trackers),
            'next_track_id': self.next_id,
            'tracks': {}
        }
        
        for track_id, track in self.trackers.items():
            info['tracks'][track_id] = {
                'class_id': int(track['class_id']),
                'age': track['last_seen'] - track['first_seen'],
                'history_length': len(track['history']),
                'current_score': float(track['score']),
            }
        
        return info
    
    def reset(self):
        """Reset tracker state."""
        self.trackers.clear()
        self.next_id = 0
        self.frame_history.clear()
        logger.info("DetectionTracker reset")