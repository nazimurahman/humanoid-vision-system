# src/training/loss_functions.py
"""
Custom loss functions for mHC vision system.

Includes:
1. MHC-enhanced YOLO loss with manifold regularization
2. Multi-task loss for hybrid vision tasks
3. Manifold constraint losses
4. Stability-preserving losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.spatial.distance import cdist

class MHCYOLOLoss(nn.Module):
    """
    YOLO loss function enhanced with manifold constraints.
    
    Features:
    - Bounding box regression with CIoU loss
    - Objectness prediction with focal loss
    - Class prediction with label smoothing
    - Manifold-aware regularization
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        anchors: Optional[List[Tuple[float, float]]] = None,
        lambda_coord: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_cls: float = 1.0,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        ciou: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        
        # Default anchors (COCO)
        if anchors is None:
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],      # Small
                [(30, 61), (62, 45), (59, 119)],     # Medium
                [(116, 90), (156, 198), (373, 326)]  # Large
            ]
        else:
            self.anchors = anchors
        
        # Loss components
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.ciou = ciou
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MHC-enhanced YOLO loss.
        
        Args:
            predictions: Dictionary of predictions from model
            targets: Dictionary of target tensors
            
        Returns:
            Dictionary of loss components
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Process each scale
        for scale_idx, scale_name in enumerate(['small_scale', 'medium_scale', 'large_scale']):
            if scale_name not in predictions:
                continue
            
            pred = predictions[scale_name]
            target = targets.get(scale_name)
            
            if target is None:
                continue
            
            # Compute losses for this scale
            scale_loss_dict = self._compute_scale_loss(pred, target, scale_idx)
            
            # Add to total loss
            for key, loss in scale_loss_dict.items():
                total_loss += loss
                loss_dict[f'{scale_name}_{key}'] = loss
        
        # Add manifold regularization if present in predictions
        if 'manifold_reg' in predictions:
            manifold_loss = predictions['manifold_reg'].mean()
            total_loss += manifold_loss * 0.01  # Small weight
            loss_dict['manifold_reg'] = manifold_loss
        
        loss_dict['total'] = total_loss
        return loss_dict
    
    def _compute_scale_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        scale_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a specific scale.
        
        Args:
            pred: Predictions [B, A, H, W, 5+C]
            target: Targets [B, A, H, W, 5+C]
            scale_idx: Scale index for anchor selection
            
        Returns:
            Dictionary of loss components
        """
        device = pred.device
        batch_size, num_anchors, height, width, _ = pred.shape
        
        # Get anchors for this scale
        anchors = torch.tensor(self.anchors[scale_idx], device=device)
        anchors = anchors.view(1, num_anchors, 1, 1, 2)
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=2)
        grid = grid.view(1, 1, height, width, 2)
        
        # Split predictions
        pred_xy = torch.sigmoid(pred[..., 0:2])  # Center offsets
        pred_wh = torch.exp(pred[..., 2:4]) * anchors  # Width/height
        pred_obj = torch.sigmoid(pred[..., 4:5])  # Objectness
        pred_cls = torch.sigmoid(pred[..., 5:])   # Class probabilities
        
        # Split targets
        target_xy = target[..., 0:2]
        target_wh = target[..., 2:4]
        target_obj = target[..., 4:5]
        target_cls = target[..., 5:]
        
        # Create masks
        obj_mask = target_obj > 0.5
        noobj_mask = target_obj < 0.5
        
        # ========== COORDINATE LOSS ==========
        if obj_mask.sum() > 0:
            # Convert to absolute coordinates
            pred_box = torch.cat([
                (pred_xy[obj_mask] + grid.expand_as(pred_xy)[obj_mask]) / width,
                pred_wh[obj_mask] / 416  # Normalize
            ], dim=-1)
            
            target_box = torch.cat([
                (target_xy[obj_mask] + grid.expand_as(target_xy)[obj_mask]) / width,
                target_wh[obj_mask] / 416
            ], dim=-1)
            
            if self.ciou:
                coord_loss = self._ciou_loss(pred_box, target_box)
            else:
                coord_loss = self.mse_loss(pred_box, target_box).mean()
            
            coord_loss = self.lambda_coord * coord_loss
        else:
            coord_loss = torch.tensor(0.0, device=device)
        
        # ========== OBJECTNESS LOSS ==========
        # With focal loss for class imbalance
        obj_loss = self._focal_loss(pred_obj[obj_mask], target_obj[obj_mask])
        obj_loss = self.lambda_obj * obj_loss.sum() / max(obj_mask.sum().item(), 1)
        
        noobj_loss = self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
        noobj_loss = self.lambda_noobj * noobj_loss.mean()
        
        # ========== CLASSIFICATION LOSS ==========
        if obj_mask.sum() > 0:
            # Apply label smoothing
            if self.label_smoothing > 0:
                target_cls_smooth = target_cls[obj_mask] * (1 - self.label_smoothing)
                target_cls_smooth = target_cls_smooth + self.label_smoothing / self.num_classes
            else:
                target_cls_smooth = target_cls[obj_mask]
            
            cls_loss = self.bce_loss(pred_cls[obj_mask], target_cls_smooth)
            cls_loss = self.lambda_cls * cls_loss.mean()
        else:
            cls_loss = torch.tensor(0.0, device=device)
        
        return {
            'coord_loss': coord_loss,
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'cls_loss': cls_loss
        }
    
    def _ciou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Complete IoU loss with aspect ratio consistency.
        
        Args:
            pred_boxes: Predicted boxes [N, 4] (x, y, w, h)
            target_boxes: Target boxes [N, 4] (x, y, w, h)
            
        Returns:
            CIoU loss
        """
        # Convert to corner format
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_width * inter_height
        
        # Union
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union = pred_area + target_area - intersection + eps
        
        # IoU
        iou = intersection / union
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_width = enclose_x2 - enclose_x1
        enclose_height = enclose_y2 - enclose_y1
        enclose_diag = enclose_width ** 2 + enclose_height ** 2 + eps
        
        # Center distance
        center_dist = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                      (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
        
        # Aspect ratio
        v = (4 / (np.pi ** 2)) * torch.pow(
            torch.atan(target_boxes[:, 2] / target_boxes[:, 3]) -
            torch.atan(pred_boxes[:, 2] / pred_boxes[:, 3]), 2
        )
        
        alpha = v / (1 - iou + v + eps)
        
        # CIoU
        ciou = iou - (center_dist / enclose_diag) - alpha * v
        
        return (1 - ciou).mean()
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.
        
        Args:
            pred: Predictions
            target: Targets
            alpha: Balancing parameter
            
        Returns:
            Focal loss
        """
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        pt = torch.exp(-bce_loss)  # prevents nans
        focal_weight = alpha * (1 - pt) ** self.focal_gamma
        
        return focal_weight * bce_loss

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for hybrid vision system.
    
    Combines:
    - Object detection loss
    - Classification loss
    - Manifold regularization loss
    - Consistency losses between scales
    """
    
    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        manifold_weight: float = 0.01,
        consistency_weight: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        # Default task weights
        if task_weights is None:
            task_weights = {
                'detection': 1.0,
                'classification': 0.5,
                'segmentation': 0.3
            }
        
        self.task_weights = task_weights
        self.manifold_weight = manifold_weight
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        
        # Individual loss functions
        self.detection_loss = MHCYOLOLoss()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.segmentation_loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
            
        Returns:
            Dictionary of loss components
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Detection loss
        if 'detections' in predictions and 'detections' in targets:
            det_loss_dict = self.detection_loss(
                predictions['detections'],
                targets['detections']
            )
            det_loss = det_loss_dict['total']
            total_loss += self.task_weights.get('detection', 1.0) * det_loss
            loss_dict['detection_loss'] = det_loss
            
            # Add individual components
            for key, value in det_loss_dict.items():
                if key != 'total':
                    loss_dict[f'detection_{key}'] = value
        
        # Classification loss
        if 'classifications' in predictions and 'labels' in targets:
            cls_loss = self.classification_loss(
                predictions['classifications'],
                targets['labels']
            )
            total_loss += self.task_weights.get('classification', 0.5) * cls_loss
            loss_dict['classification_loss'] = cls_loss
        
        # Segmentation loss (optional)
        if 'segmentations' in predictions and 'masks' in targets:
            seg_loss = self.segmentation_loss(
                predictions['segmentations'],
                targets['masks']
            )
            total_loss += self.task_weights.get('segmentation', 0.3) * seg_loss
            loss_dict['segmentation_loss'] = seg_loss
        
        # Manifold regularization
        if 'manifold_features' in predictions:
            manifold_loss = self._compute_manifold_loss(predictions['manifold_features'])
            total_loss += self.manifold_weight * manifold_loss
            loss_dict['manifold_loss'] = manifold_loss
        
        # Multi-scale consistency
        if 'multi_scale_features' in predictions:
            consistency_loss = self._compute_consistency_loss(predictions['multi_scale_features'])
            total_loss += self.consistency_weight * consistency_loss
            loss_dict['consistency_loss'] = consistency_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def _compute_manifold_loss(self, manifold_features: torch.Tensor) -> torch.Tensor:
        """
        Compute manifold regularization loss.
        
        Encourages features to lie on a well-behaved manifold.
        """
        # 1. Local linearity constraint
        B, C, H, W = manifold_features.shape
        features_flat = manifold_features.flatten(2)  # [B, C, H*W]
        
        # Compute covariance
        features_mean = features_flat.mean(dim=2, keepdim=True)
        features_centered = features_flat - features_mean
        
        # Covariance matrix
        cov_matrix = torch.bmm(features_centered, features_centered.transpose(1, 2))
        cov_matrix = cov_matrix / (H * W - 1)
        
        # Encourage well-conditioned covariance
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        condition_loss = (eigenvalues.max(dim=1)[0] / (eigenvalues.min(dim=1)[0] + 1e-8)).mean()
        
        # 2. Smoothness constraint (neighbor similarity)
        smoothness_loss = self._compute_smoothness_loss(manifold_features)
        
        return condition_loss + 0.1 * smoothness_loss
    
    def _compute_smoothness_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Encourage smoothness in feature maps."""
        # Compute gradients
        grad_x = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        grad_y = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
        
        return grad_x.mean() + grad_y.mean()
    
    def _compute_consistency_loss(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute consistency loss between different scales.
        
        Encourages consistent predictions across scales.
        """
        if len(multi_scale_features) < 2:
            return torch.tensor(0.0, device=multi_scale_features[0].device)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(multi_scale_features)):
            for j in range(i + 1, len(multi_scale_features)):
                feat_i = multi_scale_features[i]
                feat_j = multi_scale_features[j]
                
                # Resize to common size
                if feat_i.shape[-2:] != feat_j.shape[-2:]:
                    feat_j_resized = F.interpolate(
                        feat_j,
                        size=feat_i.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    feat_j_resized = feat_j
                
                # KL divergence between feature distributions
                feat_i_norm = F.softmax(feat_i.flatten(1) / self.temperature, dim=1)
                feat_j_norm = F.softmax(feat_j_resized.flatten(1) / self.temperature, dim=1)
                
                consistency_loss = F.kl_div(
                    feat_i_norm.log(),
                    feat_j_norm,
                    reduction='batchmean'
                )
                
                total_loss += consistency_loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

class ManifoldRegularizationLoss(nn.Module):
    """
    Specific loss for enforcing manifold constraints in MHC layers.
    
    Ensures:
    1. Doubly stochastic constraints
    2. Eigenvalue bounds
    3. Smooth weight updates
    """
    
    def __init__(
        self,
        double_stochastic_weight: float = 1.0,
        eigenvalue_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        target_eigenvalue_max: float = 1.0
    ):
        super().__init__()
        
        self.double_stochastic_weight = double_stochastic_weight
        self.eigenvalue_weight = eigenvalue_weight
        self.smoothness_weight = smoothness_weight
        self.target_eigenvalue_max = target_eigenvalue_max
    
    def forward(
        self,
        h_res_matrices: List[torch.Tensor],
        h_pre_matrices: List[torch.Tensor],
        h_post_matrices: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute manifold regularization losses.
        
        Args:
            h_res_matrices: List of H_res matrices from MHC layers
            h_pre_matrices: List of H_pre matrices
            h_post_matrices: List of H_post matrices
            
        Returns:
            Dictionary of regularization losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Process each MHC layer
        for i, (h_res, h_pre, h_post) in enumerate(zip(
            h_res_matrices, h_pre_matrices, h_post_matrices
        )):
            layer_losses = {}
            
            # 1. Doubly stochastic constraint for H_res
            if h_res is not None:
                row_sum = h_res.sum(dim=1)
                col_sum = h_res.sum(dim=0)
                
                row_loss = F.mse_loss(row_sum, torch.ones_like(row_sum))
                col_loss = F.mse_loss(col_sum, torch.ones_like(col_sum))
                
                double_stochastic_loss = (row_loss + col_loss) / 2
                layer_losses['double_stochastic'] = double_stochastic_loss
                total_loss += self.double_stochastic_weight * double_stochastic_loss
            
            # 2. Eigenvalue constraint
            if h_res is not None:
                eigenvalues = torch.linalg.eigvalsh(h_res)
                eigenvalue_loss = F.relu(eigenvalues - self.target_eigenvalue_max).mean()
                layer_losses['eigenvalue'] = eigenvalue_loss
                total_loss += self.eigenvalue_weight * eigenvalue_loss
            
            # 3. Smoothness constraint for H_pre and H_post
            if h_pre is not None:
                h_pre_smooth = self._compute_smoothness(h_pre)
                layer_losses['h_pre_smooth'] = h_pre_smooth
                total_loss += self.smoothness_weight * h_pre_smooth
            
            if h_post is not None:
                h_post_smooth = self._compute_smoothness(h_post)
                layer_losses['h_post_smooth'] = h_post_smooth
                total_loss += self.smoothness_weight * h_post_smooth
            
            # Add layer-specific losses to dictionary
            for key, loss in layer_losses.items():
                loss_dict[f'layer_{i}_{key}'] = loss
        
        loss_dict['total'] = total_loss
        return loss_dict
    
    def _compute_smoothness(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute smoothness of weight matrix."""
        if len(matrix.shape) == 2:
            # For 2D matrices, encourage nearby rows/columns to be similar
            row_diff = matrix[1:, :] - matrix[:-1, :]
            col_diff = matrix[:, 1:] - matrix[:, :-1]
            return row_diff.abs().mean() + col_diff.abs().mean()
        else:
            return torch.tensor(0.0, device=matrix.device)