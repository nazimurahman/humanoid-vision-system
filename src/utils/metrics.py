# src/utils/metrics.py
"""
Comprehensive Metrics for Vision System Evaluation.

This module provides:
1. Object detection metrics (mAP, IoU, precision, recall)
2. Classification metrics (accuracy, F1, confusion matrix)
3. Training stability metrics (gradient norms, eigenvalue monitoring)
4. Inference performance metrics (latency, throughput, memory)
5. Manifold constraint metrics (Sinkhorn convergence, constraint violations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from collections import defaultdict
import time
import math

from .sinkhorn import SinkhornKnopp
from .manifold_ops import check_manifold_constraints

@dataclass
class DetectionMetrics:
    """Container for detection metrics."""
    mAP_50: float = 0.0
    mAP_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    average_iou: float = 0.0
    num_detections: int = 0
    num_ground_truth: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mAP_50': self.mAP_50,
            'mAP_50_95': self.mAP_50_95,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'average_iou': self.average_iou,
            'num_detections': self.num_detections,
            'num_ground_truth': self.num_ground_truth,
        }


class DetectionEvaluator:
    """
    Evaluator for object detection tasks.
    
    Implements COCO-style evaluation metrics including:
    - Mean Average Precision (mAP) at different IoU thresholds
    - Precision-Recall curves
    - Per-class metrics
    - Inference speed metrics
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        iou_thresholds: List[float] = None,
        conf_threshold: float = 0.5,
        max_detections: int = 100,
        use_coco_metric: bool = True
    ):
        """
        Initialize detection evaluator.
        
        Args:
            num_classes: Number of object classes
            iou_thresholds: IoU thresholds for mAP calculation
            conf_threshold: Confidence threshold for detections
            max_detections: Maximum detections per image
            use_coco_metric: Use COCO evaluation protocol
        """
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        
        if iou_thresholds is None:
            if use_coco_metric:
                # COCO standard: 10 IoU thresholds from 0.5 to 0.95
                self.iou_thresholds = np.linspace(0.5, 0.95, 10)
            else:
                # Pascal VOC standard: single IoU 0.5
                self.iou_thresholds = [0.5]
        else:
            self.iou_thresholds = iou_thresholds
        
        # Storage for evaluation
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        # Per-class storage
        self.detections = [[] for _ in range(self.num_classes)]
        self.ground_truths = [[] for _ in range(self.num_classes)]
        
        # Metrics storage
        self.results = {}
        self.per_class_metrics = {}
        
        # Timing
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
    
    def add_batch(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        inference_time: Optional[float] = None,
        preprocessing_time: Optional[float] = None,
        postprocessing_time: Optional[float] = None
    ):
        """
        Add a batch of predictions and ground truths.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            inference_time: Inference time for the batch
            preprocessing_time: Preprocessing time
            postprocessing_time: Postprocessing time
        """
        assert len(predictions) == len(ground_truths), \
            "Number of predictions must match number of ground truths"
        
        # Store timing information
        if inference_time is not None:
            self.inference_times.append(inference_time)
        if preprocessing_time is not None:
            self.preprocessing_times.append(preprocessing_time)
        if postprocessing_time is not None:
            self.postprocessing_times.append(postprocessing_time)
        
        # Process each image
        for pred, gt in zip(predictions, ground_truths):
            self._process_single_image(pred, gt)
    
    def _process_single_image(self, pred: Dict, gt: Dict):
        """
        Process predictions and ground truths for a single image.
        
        Args:
            pred: Prediction dictionary with keys:
                - boxes: [N, 4] tensor of bounding boxes
                - scores: [N] tensor of confidence scores
                - labels: [N] tensor of class labels
            gt: Ground truth dictionary with keys:
                - boxes: [M, 4] tensor of bounding boxes
                - labels: [M] tensor of class labels
        """
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
        pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
        pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
        
        gt_boxes = gt['boxes'].cpu().numpy() if torch.is_tensor(gt['boxes']) else gt['boxes']
        gt_labels = gt['labels'].cpu().numpy() if torch.is_tensor(gt['labels']) else gt['labels']
        
        # Apply confidence threshold
        keep = pred_scores >= self.conf_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        
        # Limit number of detections
        if len(pred_boxes) > self.max_detections:
            # Sort by confidence and keep top-k
            top_k = np.argsort(pred_scores)[-self.max_detections:]
            pred_boxes = pred_boxes[top_k]
            pred_scores = pred_scores[top_k]
            pred_labels = pred_labels[top_k]
        
        # Process per class
        for class_id in range(self.num_classes):
            # Get predictions for this class
            class_mask = pred_labels == class_id
            if class_mask.any():
                self.detections[class_id].append({
                    'boxes': pred_boxes[class_mask],
                    'scores': pred_scores[class_mask],
                    'image_id': len(self.detections[class_id])
                })
            
            # Get ground truths for this class
            gt_class_mask = gt_labels == class_id
            if gt_class_mask.any():
                self.ground_truths[class_id].append({
                    'boxes': gt_boxes[gt_class_mask],
                    'image_id': len(self.ground_truths[class_id])
                })
    
    def compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: [N, 4] array of boxes (x1, y1, x2, y2)
            boxes2: [M, 4] array of boxes
            
        Returns:
            IoU matrix of shape [N, M]
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1[:, None, :]  # [N, 1, 4]
        boxes2 = boxes2[None, :, :]  # [1, M, 4]
        
        # Compute intersection
        inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        inter_width = np.maximum(0, inter_x2 - inter_x1)
        inter_height = np.maximum(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        
        # Compute union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def compute_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        Compute Average Precision using the precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            
        Returns:
            Average Precision
        """
        # Append sentinel values
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        
        # Compute the precision envelope
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # Find indices where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Compute AP as the area under the PR curve
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap
    
    def evaluate_class(self, class_id: int) -> Dict[str, float]:
        """
        Evaluate metrics for a single class.
        
        Args:
            class_id: Class ID to evaluate
            
        Returns:
            Dictionary of metrics for the class
        """
        detections = self.detections[class_id]
        ground_truths = self.ground_truths[class_id]
        
        if not detections or not ground_truths:
            return {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_detections': 0,
                'num_ground_truth': sum(len(gt['boxes']) for gt in ground_truths)
            }
        
        # Flatten detections and ground truths
        all_detections = []
        all_scores = []
        all_image_ids = []
        
        for det in detections:
            for i, (box, score) in enumerate(zip(det['boxes'], det['scores'])):
                all_detections.append(box)
                all_scores.append(score)
                all_image_ids.append(det['image_id'])
        
        all_detections = np.array(all_detections)
        all_scores = np.array(all_scores)
        all_image_ids = np.array(all_image_ids)
        
        # Sort by confidence
        sort_idx = np.argsort(-all_scores)
        all_detections = all_detections[sort_idx]
        all_scores = all_scores[sort_idx]
        all_image_ids = all_image_ids[sort_idx]
        
        # Initialize matching arrays
        num_detections = len(all_detections)
        num_ground_truth = sum(len(gt['boxes']) for gt in ground_truths)
        
        tp = np.zeros(num_detections)
        fp = np.zeros(num_detections)
        
        # Create dictionary for ground truths per image
        gt_by_image = {}
        for gt in ground_truths:
            gt_by_image[gt['image_id']] = gt['boxes']
        
        # Match detections to ground truths
        detected_gt = {}
        
        for i in range(num_detections):
            image_id = all_image_ids[i]
            detection = all_detections[i]
            
            if image_id not in gt_by_image:
                fp[i] = 1
                continue
            
            gt_boxes = gt_by_image[image_id]
            
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            # Compute IoU with all ground truths in this image
            ious = self.compute_iou(detection[None, :], gt_boxes)
            max_iou = np.max(ious)
            best_gt_idx = np.argmax(ious)
            
            # Check if this ground truth has already been detected
            gt_key = (image_id, best_gt_idx)
            
            if max_iou >= self.iou_thresholds[0] and gt_key not in detected_gt:
                tp[i] = 1
                detected_gt[gt_key] = True
            else:
                fp[i] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall_curve = tp_cumsum / (num_ground_truth + 1e-6)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP
        ap = self.compute_ap(precision_curve, recall_curve)
        
        # Compute final precision and recall
        precision = precision_curve[-1] if len(precision_curve) > 0 else 0.0
        recall = recall_curve[-1] if len(recall_curve) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_detections': num_detections,
            'num_ground_truth': num_ground_truth,
            'tp': tp_cumsum[-1] if len(tp_cumsum) > 0 else 0,
            'fp': fp_cumsum[-1] if len(fp_cumsum) > 0 else 0,
        }
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        results = {}
        
        # Compute per-class metrics
        class_metrics = []
        for class_id in range(self.num_classes):
            metrics = self.evaluate_class(class_id)
            class_metrics.append(metrics)
        
        # Compute mAP at different IoU thresholds
        map_metrics = {}
        for iou_thresh in self.iou_thresholds:
            # Temporarily set IoU threshold and recompute
            original_threshold = self.iou_thresholds[0]
            self.iou_thresholds = [iou_thresh]
            
            aps = []
            for class_id in range(self.num_classes):
                metrics = self.evaluate_class(class_id)
                aps.append(metrics['ap'])
            
            map_metrics[f'mAP_{iou_thresh:.2f}'] = np.mean(aps) if aps else 0.0
            
            # Restore original threshold
            self.iou_thresholds = [original_threshold]
        
        # Compute overall mAP (average over IoU thresholds)
        if len(self.iou_thresholds) > 1:
            results['mAP_50_95'] = np.mean(list(map_metrics.values()))
        
        # Get mAP at 0.5 IoU
        results['mAP_50'] = map_metrics.get('mAP_0.50', 0.0)
        
        # Compute overall precision, recall, F1
        all_tp = sum(m['tp'] for m in class_metrics)
        all_fp = sum(m['fp'] for m in class_metrics)
        all_fn = sum(m['num_ground_truth'] - m['tp'] for m in class_metrics)
        
        results['precision'] = all_tp / (all_tp + all_fp + 1e-6)
        results['recall'] = all_tp / (all_tp + all_fn + 1e-6)
        results['f1_score'] = (
            2 * results['precision'] * results['recall'] / 
            (results['precision'] + results['recall'] + 1e-6)
        )
        
        # Store per-class metrics
        results['per_class'] = class_metrics
        
        # Compute inference performance metrics
        if self.inference_times:
            results['inference_time_mean'] = np.mean(self.inference_times)
            results['inference_time_std'] = np.std(self.inference_times)
            results['inference_fps'] = 1.0 / results['inference_time_mean']
        
        if self.preprocessing_times:
            results['preprocessing_time_mean'] = np.mean(self.preprocessing_times)
        
        if self.postprocessing_times:
            results['postprocessing_time_mean'] = np.mean(self.postprocessing_times)
        
        self.results = results
        return results
    
    def get_summary(self) -> str:
        """Get formatted summary string."""
        if not self.results:
            self.compute()
        
        summary_lines = [
            "=" * 60,
            "Detection Metrics Summary",
            "=" * 60,
            f"mAP@0.50: {self.results.get('mAP_50', 0):.4f}",
            f"mAP@0.50:0.95: {self.results.get('mAP_50_95', 0):.4f}",
            f"Precision: {self.results.get('precision', 0):.4f}",
            f"Recall: {self.results.get('recall', 0):.4f}",
            f"F1 Score: {self.results.get('f1_score', 0):.4f}",
        ]
        
        if 'inference_fps' in self.results:
            summary_lines.append(f"Inference FPS: {self.results['inference_fps']:.2f}")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)


class StabilityMetrics:
    """
    Monitor training stability metrics for mHC-based models.
    
    Tracks:
    - Gradient norms and explosions
    - Eigenvalue spectra of constrained matrices
    - Sinkhorn convergence
    - Signal propagation through layers
    - Loss landscape smoothness
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize stability metrics tracker.
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        
        # Gradient monitoring
        self.gradient_norms = []
        self.gradient_means = []
        self.gradient_stds = []
        
        # Eigenvalue monitoring
        self.eigenvalue_spectra = []
        self.condition_numbers = []
        
        # Sinkhorn convergence
        self.sinkhorn_errors = []
        self.sinkhorn_iterations = []
        
        # Signal propagation
        self.activation_norms = []
        self.weight_updates = []
        
        # Loss landscape
        self.loss_values = []
        self.loss_gradients = []
        
        # Constraint violations
        self.constraint_violations = {
            'birkhoff': [],
            'stiefel': [],
            'spd': []
        }
    
    def update_gradient_metrics(self, model: nn.Module):
        """
        Update gradient-related metrics.
        
        Args:
            model: PyTorch model
        """
        grad_norms = []
        grad_means = []
        grad_stds = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_norms.append(grad.norm().item())
                grad_means.append(grad.mean().item())
                grad_stds.append(grad.std().item())
        
        if grad_norms:
            self.gradient_norms.append(np.mean(grad_norms))
            self.gradient_means.append(np.mean(grad_means))
            self.gradient_stds.append(np.mean(grad_stds))
            
            # Keep window size
            self._trim_to_window(self.gradient_norms)
            self._trim_to_window(self.gradient_means)
            self._trim_to_window(self.gradient_stds)
    
    def update_eigenvalue_metrics(self, matrices: Dict[str, torch.Tensor]):
        """
        Update eigenvalue-based stability metrics.
        
        Args:
            matrices: Dictionary of matrices to analyze
        """
        for name, matrix in matrices.items():
            if matrix is None or matrix.numel() == 0:
                continue
            
            try:
                # Ensure matrix is 2D
                if matrix.dim() > 2:
                    matrix = matrix.reshape(-1, matrix.shape[-1])
                
                # Compute eigenvalues
                if matrix.shape[0] == matrix.shape[1]:
                    eigenvalues = torch.linalg.eigvalsh(matrix).cpu().numpy()
                    
                    # Store eigenvalue spectrum
                    self.eigenvalue_spectra.append({
                        'name': name,
                        'eigenvalues': eigenvalues,
                        'max_eigenvalue': np.max(eigenvalues),
                        'min_eigenvalue': np.min(eigenvalues),
                        'condition_number': np.max(np.abs(eigenvalues)) / (np.min(np.abs(eigenvalues)) + 1e-6)
                    })
                    
                    self.condition_numbers.append(self.eigenvalue_spectra[-1]['condition_number'])
            except Exception as e:
                warnings.warn(f"Eigenvalue computation failed for {name}: {e}")
        
        # Keep window size
        self._trim_to_window(self.condition_numbers)
    
    def update_sinkhorn_metrics(self, projector: SinkhornKnopp):
        """
        Update Sinkhorn convergence metrics.
        
        Args:
            projector: SinkhornKnopp projector
        """
        diagnostics = projector.get_diagnostics()
        
        if diagnostics:
            self.sinkhorn_errors.append(
                diagnostics.get('sinkhorn_row_error_mean', 0) +
                diagnostics.get('sinkhorn_col_error_mean', 0)
            )
            self.sinkhorn_iterations.append(
                diagnostics.get('sinkhorn_iterations', 0)
            )
            
            # Keep window size
            self._trim_to_window(self.sinkhorn_errors)
            self._trim_to_window(self.sinkhorn_iterations)
    
    def update_constraint_violations(
        self,
        matrices: Dict[str, torch.Tensor],
        manifold_type: str = 'birkhoff'
    ):
        """
        Update constraint violation metrics.
        
        Args:
            matrices: Dictionary of matrices to check
            manifold_type: Type of manifold constraints
        """
        violations = []
        
        for name, matrix in matrices.items():
            checks = check_manifold_constraints(matrix, manifold_type)
            
            if 'row_error' in checks and 'col_error' in checks:
                violation = checks['row_error'] + checks['col_error']
                violations.append(violation)
        
        if violations:
            self.constraint_violations[manifold_type].append(np.mean(violations))
            self._trim_to_window(self.constraint_violations[manifold_type])
    
    def update_activation_metrics(self, activations: Dict[str, torch.Tensor]):
        """
        Update activation norm metrics.
        
        Args:
            activations: Dictionary of layer activations
        """
        norms = []
        
        for name, activation in activations.items():
            if activation is not None:
                norm = activation.norm().item()
                norms.append(norm)
        
        if norms:
            self.activation_norms.append(np.mean(norms))
            self._trim_to_window(self.activation_norms)
    
    def update_loss_metrics(self, loss: float, loss_grad: Optional[float] = None):
        """
        Update loss-related metrics.
        
        Args:
            loss: Current loss value
            loss_grad: Loss gradient norm (optional)
        """
        self.loss_values.append(loss)
        if loss_grad is not None:
            self.loss_gradients.append(loss_grad)
        
        self._trim_to_window(self.loss_values)
        if self.loss_gradients:
            self._trim_to_window(self.loss_gradients)
    
    def _trim_to_window(self, array: list):
        """Trim array to window size."""
        if len(array) > self.window_size:
            del array[:-self.window_size]
    
    def get_stability_score(self) -> float:
        """
        Compute overall stability score.
        
        Returns:
            Stability score between 0 (unstable) and 1 (stable)
        """
        scores = []
        
        # Gradient stability (should not explode/vanish)
        if self.gradient_norms:
            grad_score = self._compute_gradient_stability()
            scores.append(grad_score)
        
        # Eigenvalue stability (should be near 1 for doubly stochastic)
        if self.condition_numbers:
            eigen_score = self._compute_eigenvalue_stability()
            scores.append(eigen_score)
        
        # Sinkhorn convergence (errors should be small)
        if self.sinkhorn_errors:
            sinkhorn_score = self._compute_sinkhorn_stability()
            scores.append(sinkhorn_score)
        
        # Constraint satisfaction
        for manifold, violations in self.constraint_violations.items():
            if violations:
                constraint_score = self._compute_constraint_stability(violations)
                scores.append(constraint_score)
        
        # Activation stability (should not explode)
        if self.activation_norms:
            activation_score = self._compute_activation_stability()
            scores.append(activation_score)
        
        if not scores:
            return 1.0  # Default to stable if no metrics
        
        return np.mean(scores)
    
    def _compute_gradient_stability(self) -> float:
        """Compute gradient stability score."""
        if len(self.gradient_norms) < 2:
            return 1.0
        
        # Check for gradient explosion/vanishing
        recent_norms = self.gradient_norms[-10:]
        mean_norm = np.mean(recent_norms)
        
        # Ideal gradient norm is around 1.0
        if mean_norm > 100:  # Exploding gradients
            return 0.0
        elif mean_norm < 0.01:  # Vanishing gradients
            return 0.0
        else:
            # Score based on distance from 1.0
            return 1.0 / (1.0 + abs(np.log10(mean_norm)))
    
    def _compute_eigenvalue_stability(self) -> float:
        """Compute eigenvalue stability score."""
        if not self.eigenvalue_spectra:
            return 1.0
        
        # Check recent eigenvalue spectra
        recent_spectra = self.eigenvalue_spectra[-5:]
        
        scores = []
        for spectrum in recent_spectra:
            eigenvalues = spectrum['eigenvalues']
            
            # For doubly stochastic matrices, eigenvalues should be in [0, 1]
            max_eig = spectrum['max_eigenvalue']
            min_eig = spectrum['min_eigenvalue']
            
            # Score based on how close eigenvalues are to [0, 1]
            if max_eig > 1.1 or min_eig < -0.1:
                score = 0.0
            else:
                # Distance from ideal [0, 1] range
                max_dist = max(0, max_eig - 1.0, -min_eig)
                score = 1.0 / (1.0 + 10 * max_dist)
            
            scores.append(score)
        
        return np.mean(scores) if scores else 1.0
    
    def _compute_sinkhorn_stability(self) -> float:
        """Compute Sinkhorn convergence stability score."""
        if not self.sinkhorn_errors:
            return 1.0
        
        recent_errors = self.sinkhorn_errors[-10:]
        mean_error = np.mean(recent_errors)
        
        # Score based on error magnitude
        if mean_error > 0.1:  # Large error
            return 0.0
        elif mean_error > 0.01:  # Moderate error
            return 0.5
        else:  # Small error
            return 1.0
    
    def _compute_constraint_stability(self, violations: list) -> float:
        """Compute constraint satisfaction score."""
        if not violations:
            return 1.0
        
        recent_violations = violations[-10:]
        mean_violation = np.mean(recent_violations)
        
        # Score based on violation magnitude
        return 1.0 / (1.0 + 100 * mean_violation)
    
    def _compute_activation_stability(self) -> float:
        """Compute activation stability score."""
        if len(self.activation_norms) < 2:
            return 1.0
        
        # Check for activation explosion
        recent_norms = self.activation_norms[-10:]
        mean_norm = np.mean(recent_norms)
        
        # Large activation norms indicate instability
        if mean_norm > 1000:
            return 0.0
        elif mean_norm > 100:
            return 0.5
        else:
            return 1.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all computed metrics
        """
        summary = {
            'stability_score': self.get_stability_score(),
            'gradient_metrics': {},
            'eigenvalue_metrics': {},
            'sinkhorn_metrics': {},
            'constraint_metrics': {},
            'activation_metrics': {},
            'loss_metrics': {},
        }
        
        # Gradient metrics
        if self.gradient_norms:
            summary['gradient_metrics'].update({
                'mean_norm': np.mean(self.gradient_norms[-10:]),
                'std_norm': np.std(self.gradient_norms[-10:]),
                'max_norm': np.max(self.gradient_norms[-10:]),
                'min_norm': np.min(self.gradient_norms[-10:]),
            })
        
        # Eigenvalue metrics
        if self.eigenvalue_spectra:
            recent_spectra = self.eigenvalue_spectra[-5:]
            summary['eigenvalue_metrics'].update({
                'mean_condition_number': np.mean(self.condition_numbers[-10:]) if self.condition_numbers else 0,
                'num_spectra_analyzed': len(recent_spectra),
            })
        
        # Sinkhorn metrics
        if self.sinkhorn_errors:
            summary['sinkhorn_metrics'].update({
                'mean_error': np.mean(self.sinkhorn_errors[-10:]),
                'mean_iterations': np.mean(self.sinkhorn_iterations[-10:]) if self.sinkhorn_iterations else 0,
            })
        
        # Constraint metrics
        for manifold, violations in self.constraint_violations.items():
            if violations:
                summary['constraint_metrics'][manifold] = {
                    'mean_violation': np.mean(violations[-10:]),
                    'max_violation': np.max(violations[-10:]),
                }
        
        # Activation metrics
        if self.activation_norms:
            summary['activation_metrics'].update({
                'mean_norm': np.mean(self.activation_norms[-10:]),
                'std_norm': np.std(self.activation_norms[-10:]),
            })
        
        # Loss metrics
        if self.loss_values:
            summary['loss_metrics'].update({
                'mean_loss': np.mean(self.loss_values[-10:]),
                'loss_std': np.std(self.loss_values[-10:]),
            })
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.gradient_norms.clear()
        self.gradient_means.clear()
        self.gradient_stds.clear()
        self.eigenvalue_spectra.clear()
        self.condition_numbers.clear()
        self.sinkhorn_errors.clear()
        self.sinkhorn_iterations.clear()
        self.activation_norms.clear()
        self.weight_updates.clear()
        self.loss_values.clear()
        self.loss_gradients.clear()
        
        for key in self.constraint_violations:
            self.constraint_violations[key].clear()


class InferenceMetrics:
    """
    Metrics for inference performance evaluation.
    
    Tracks:
    - Latency (preprocessing, inference, postprocessing)
    - Throughput (FPS)
    - Memory usage (GPU/CPU)
    - Hardware utilization
    - Quality of Service metrics
    """
    
    def __init__(self):
        """Initialize inference metrics tracker."""
        self.timestamps = {
            'preprocess': [],
            'inference': [],
            'postprocess': [],
            'total': []
        }
        
        self.memory_usage = {
            'gpu_memory': [],
            'cpu_memory': [],
            'gpu_utilization': [],
            'cpu_utilization': []
        }
        
        self.throughput_metrics = {
            'fps': [],
            'batch_times': [],
            'queue_lengths': []
        }
        
        self.quality_metrics = {
            'success_rate': [],
            'timeout_count': 0,
            'error_count': 0
        }
        
        self.hardware_info = {}
    
    def start_timer(self, stage: str):
        """
        Start timer for a specific stage.
        
        Args:
            stage: Stage name ('preprocess', 'inference', 'postprocess', 'total')
        """
        if stage not in self.timestamps:
            self.timestamps[stage] = []
        
        # Store start time as tuple (start_time, None)
        self.timestamps[stage].append((time.time(), None))
    
    def stop_timer(self, stage: str):
        """
        Stop timer for a specific stage.
        
        Args:
            stage: Stage name
        """
        if stage in self.timestamps and self.timestamps[stage]:
            # Get the last started timer
            start_time, _ = self.timestamps[stage][-1]
            elapsed = time.time() - start_time
            
            # Update with elapsed time
            self.timestamps[stage][-1] = (start_time, elapsed)
            
            return elapsed
        
        return 0.0
    
    def update_memory_metrics(self):
        """Update memory usage metrics."""
        try:
            # GPU memory (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                
                self.memory_usage['gpu_memory'].append(gpu_memory)
                self.memory_usage['gpu_utilization'].append(gpu_util)
            
            # CPU memory (requires psutil)
            try:
                import psutil
                process = psutil.Process()
                cpu_memory = process.memory_info().rss / 1024**3  # GB
                cpu_util = process.cpu_percent() / 100.0  # Normalize to [0, 1]
                
                self.memory_usage['cpu_memory'].append(cpu_memory)
                self.memory_usage['cpu_utilization'].append(cpu_util)
            except ImportError:
                pass
                
        except Exception as e:
            warnings.warn(f"Failed to update memory metrics: {e}")
    
    def update_throughput(self, batch_size: int = 1):
        """
        Update throughput metrics.
        
        Args:
            batch_size: Size of processed batch
        """
        if 'total' in self.timestamps and self.timestamps['total']:
            # Get the last total time
            _, elapsed = self.timestamps['total'][-1]
            
            if elapsed and elapsed > 0:
                fps = batch_size / elapsed
                self.throughput_metrics['fps'].append(fps)
                self.throughput_metrics['batch_times'].append(elapsed)
    
    def record_success(self, success: bool = True):
        """
        Record inference success/failure.
        
        Args:
            success: Whether inference was successful
        """
        self.quality_metrics['success_rate'].append(1.0 if success else 0.0)
        
        if not success:
            self.quality_metrics['error_count'] += 1
    
    def record_timeout(self):
        """Record a timeout event."""
        self.quality_metrics['timeout_count'] += 1
        self.quality_metrics['success_rate'].append(0.0)
    
    def get_summary(self, window: int = 100) -> Dict[str, Any]:
        """
        Get inference metrics summary.
        
        Args:
            window: Window size for moving averages
            
        Returns:
            Dictionary with inference metrics
        """
        summary = {
            'latency': {},
            'throughput': {},
            'memory': {},
            'quality': {},
            'hardware': self.hardware_info.copy()
        }
        
        # Compute latency statistics
        for stage, times in self.timestamps.items():
            if times:
                # Get elapsed times (ignore incomplete measurements)
                elapsed_times = [t[1] for t in times if t[1] is not None]
                
                if elapsed_times:
                    recent_times = elapsed_times[-window:] if len(elapsed_times) > window else elapsed_times
                    
                    summary['latency'][f'{stage}_mean_ms'] = np.mean(recent_times) * 1000
                    summary['latency'][f'{stage}_std_ms'] = np.std(recent_times) * 1000
                    summary['latency'][f'{stage}_p95_ms'] = np.percentile(recent_times, 95) * 1000
                    summary['latency'][f'{stage}_p99_ms'] = np.percentile(recent_times, 99) * 1000
        
        # Compute throughput statistics
        if self.throughput_metrics['fps']:
            fps_values = self.throughput_metrics['fps'][-window:] if len(self.throughput_metrics['fps']) > window else self.throughput_metrics['fps']
            batch_times = self.throughput_metrics['batch_times'][-window:] if len(self.throughput_metrics['batch_times']) > window else self.throughput_metrics['batch_times']
            
            summary['throughput']['fps_mean'] = np.mean(fps_values)
            summary['throughput']['fps_std'] = np.std(fps_values)
            summary['throughput']['batch_time_mean_ms'] = np.mean(batch_times) * 1000
        
        # Compute memory statistics
        for metric, values in self.memory_usage.items():
            if values:
                recent_values = values[-window:] if len(values) > window else values
                summary['memory'][f'{metric}_mean'] = np.mean(recent_values)
                summary['memory'][f'{metric}_max'] = np.max(recent_values)
        
        # Compute quality statistics
        if self.quality_metrics['success_rate']:
            success_rates = self.quality_metrics['success_rate'][-window:] if len(self.quality_metrics['success_rate']) > window else self.quality_metrics['success_rate']
            
            summary['quality']['success_rate'] = np.mean(success_rates)
            summary['quality']['error_count'] = self.quality_metrics['error_count']
            summary['quality']['timeout_count'] = self.quality_metrics['timeout_count']
            summary['quality']['total_requests'] = len(self.quality_metrics['success_rate'])
        
        return summary
    
    def print_summary(self, window: int = 100):
        """Print formatted summary of inference metrics."""
        summary = self.get_summary(window)
        
        print("=" * 60)
        print("Inference Performance Summary")
        print("=" * 60)
        
        # Latency
        print("\nLatency (ms):")
        for key, value in summary['latency'].items():
            print(f"  {key}: {value:.2f}")
        
        # Throughput
        if 'throughput' in summary:
            print("\nThroughput:")
            for key, value in summary['throughput'].items():
                if 'fps' in key:
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:.2f}")
        
        # Memory
        if 'memory' in summary:
            print("\nMemory Usage:")
            for key, value in summary['memory'].items():
                if 'memory' in key:
                    print(f"  {key}: {value:.2f} GB")
                elif 'utilization' in key:
                    print(f"  {key}: {value:.1%}")
        
        # Quality
        if 'quality' in summary:
            print("\nQuality Metrics:")
            for key, value in summary['quality'].items():
                if 'rate' in key:
                    print(f"  {key}: {value:.1%}")
                else:
                    print(f"  {key}: {value}")
        
        print("=" * 60)
    
    def reset(self):
        """Reset all inference metrics."""
        for key in self.timestamps:
            self.timestamps[key].clear()
        
        for key in self.memory_usage:
            self.memory_usage[key].clear()
        
        for key in self.throughput_metrics:
            self.throughput_metrics[key].clear()
        
        self.quality_metrics['success_rate'].clear()
        self.quality_metrics['timeout_count'] = 0
        self.quality_metrics['error_count'] = 0


# Convenience functions
def compute_iou_batch(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU between two batches of boxes.
    
    Args:
        boxes1: [N, 4] tensor (x1, y1, x2, y2)
        boxes2: [M, 4] tensor
        
    Returns:
        IoU matrix of shape [N, M]
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
    
    # Compute intersection
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_width * inter_height
    
    # Compute union
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def compute_precision_recall(
    tp: np.ndarray,
    fp: np.ndarray,
    num_ground_truth: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision and recall curves.
    
    Args:
        tp: True positive array
        fp: False positive array
        num_ground_truth: Number of ground truth objects
        
    Returns:
        Tuple of (precision, recall) arrays
    """
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recall = tp_cumsum / (num_ground_truth + 1e-6)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    return precision, recall


def smooth_loss_curve(
    losses: np.ndarray,
    window_size: int = 10
) -> np.ndarray:
    """
    Smooth loss curve using moving average.
    
    Args:
        losses: Array of loss values
        window_size: Size of moving window
        
    Returns:
        Smoothed loss curve
    """
    if len(losses) < window_size:
        return losses
    
    smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    
    # Pad to original length
    pad_size = len(losses) - len(smoothed)
    padded = np.pad(smoothed, (pad_size, 0), mode='edge')
    
    return padded