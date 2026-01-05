# src/models/yolo_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional, Any
from .manifold_layers import ManifoldHyperConnection


class YOLOAnchorGenerator(nn.Module):
    """
    Anchor box generator for YOLO detection.
    
    Generates anchor boxes at different scales and aspect ratios.
    """
    
    def __init__(
        self,
        anchor_sizes: List[Tuple[int, int]] = None,
        grid_sizes: List[int] = [13, 26, 52]
    ):
        super().__init__()
        
        # Default anchor sizes (COCO dataset)
        if anchor_sizes is None:
            anchor_sizes = [
                [(10, 13), (16, 30), (33, 23)],    # Small objects
                [(30, 61), (62, 45), (59, 119)],   # Medium objects
                [(116, 90), (156, 198), (373, 326)] # Large objects
            ]
        
        self.anchor_sizes = anchor_sizes
        self.grid_sizes = grid_sizes
        self.num_scales = len(grid_sizes)
        self.num_anchors = len(anchor_sizes[0])
        
        # Register anchors as buffers
        self.register_buffer('anchors', self._generate_anchors())
    
    def _generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes for all scales."""
        all_anchors = []
        
        for scale_idx, grid_size in enumerate(self.grid_sizes):
            anchors_at_scale = []
            
            for anchor_idx, (w, h) in enumerate(self.anchor_sizes[scale_idx]):
                # Normalize anchor dimensions
                anchor_w = w / 416.0  # Assuming 416 input
                anchor_h = h / 416.0
                
                # Create grid of anchor centers
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(grid_size),
                    torch.arange(grid_size),
                    indexing='ij'
                )
                
                # Anchor format: [x_center, y_center, width, height]
                anchors = torch.stack([
                    (grid_x + 0.5) / grid_size,  # x center
                    (grid_y + 0.5) / grid_size,  # y center
                    torch.full_like(grid_x, anchor_w),  # width
                    torch.full_like(grid_x, anchor_h),  # height
                ], dim=-1)
                
                anchors_at_scale.append(anchors)
            
            # Concatenate anchors at this scale
            scale_anchors = torch.stack(anchors_at_scale, dim=0)
            all_anchors.append(scale_anchors)
        
        return torch.stack(all_anchors)  # [num_scales, num_anchors, grid, grid, 4]
    
    def forward(self, scale_idx: int) -> torch.Tensor:
        """
        Get anchors for specific scale.
        
        Args:
            scale_idx: Index of scale (0, 1, or 2)
            
        Returns:
            Anchors for this scale [num_anchors, grid, grid, 4]
        """
        return self.anchors[scale_idx]
    
    def get_num_anchors(self) -> int:
        """Get number of anchors per grid cell."""
        return self.num_anchors


class YOLOPredictionHead(nn.Module):
    """
    YOLO prediction head for object detection.
    
    Predicts:
    - Bounding box coordinates (x, y, w, h)
    - Objectness score
    - Class probabilities
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        num_anchors: int = 3,
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Each anchor predicts: 4 bbox coords + 1 objectness + num_classes
        self.output_dim = num_anchors * (5 + num_classes)
        
        # Convolutional layers for feature processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        
        # Optional MHC enhancement
        if use_mhc:
            self.mhc_enhance = ManifoldHyperConnection(
                input_dim=in_channels,
                expansion_rate=2
            )
        else:
            self.mhc_enhance = nn.Identity()
        
        # Final prediction convolution
        self.pred_conv = nn.Conv2d(
            in_channels, self.output_dim,
            kernel_size=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for YOLO head."""
        # Initialize conv layers
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Initialize prediction convolution
        nn.init.normal_(self.pred_conv.weight, std=0.01)
        nn.init.zeros_(self.pred_conv.bias)
        
        # Initialize bias for objectness predictions
        # This helps training start with reasonable predictions
        with torch.no_grad():
            bias = self.pred_conv.bias.view(self.num_anchors, -1)
            bias[:, 4] = -4.0  # Objectness bias (sigmoid(-4) ≈ 0.018)
            bias[:, 5:] = -math.log((1 - 0.01) / 0.01) / self.num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through YOLO head.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Predictions [B, num_anchors, H, W, 5+num_classes]
        """
        B, C, H, W = x.shape
        
        # Process features
        x = self.conv_layers(x)
        
        # Apply MHC enhancement
        if self.mhc_enhance is not None:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            x_enhanced = self.mhc_enhance(x_flat)
            x = x_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Final prediction
        predictions = self.pred_conv(x)  # [B, output_dim, H, W]
        
        # Reshape to separate anchors
        predictions = predictions.view(
            B, self.num_anchors, 5 + self.num_classes, H, W
        )
        
        # Permute to final format
        predictions = predictions.permute(0, 1, 3, 4, 2)  # [B, A, H, W, 5+C]
        
        return predictions


class YOLODecoder(nn.Module):
    """
    Decodes YOLO predictions to actual bounding boxes.
    
    Converts network outputs to:
    - Absolute coordinates (x, y, w, h in image space)
    - Objectness scores
    - Class probabilities
    """
    
    def __init__(self, image_size: int = 416):
        super().__init__()
        self.image_size = image_size
        
    def forward(
        self,
        predictions: torch.Tensor,
        anchors: torch.Tensor,
        grid_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Decode predictions to bounding boxes.
        
        Args:
            predictions: Raw predictions [B, A, H, W, 5+C]
            anchors: Anchor boxes [A, H, W, 4]
            grid_size: (grid_height, grid_width)
            
        Returns:
            Dictionary with decoded boxes and scores
        """
        B, A, H, W, _ = predictions.shape
        grid_h, grid_w = grid_size
        
        # Split predictions
        xy_pred = torch.sigmoid(predictions[..., 0:2])  # Center offsets
        wh_pred = predictions[..., 2:4]                 # Width/height log-space
        obj_pred = torch.sigmoid(predictions[..., 4:5]) # Objectness
        cls_pred = torch.sigmoid(predictions[..., 5:])  # Class probabilities
        
        # Create grid for cell positions
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=predictions.device),
            torch.arange(W, device=predictions.device),
            indexing='ij'
        )
        
        grid_x = grid_x.view(1, 1, H, W)
        grid_y = grid_y.view(1, 1, H, W)
        
        # Decode box coordinates
        # x = (grid_x + sigmoid(tx)) / grid_w
        # y = (grid_y + sigmoid(ty)) / grid_h
        # w = anchor_w * exp(tw) / grid_w
        # h = anchor_h * exp(th) / grid_h
        
        box_x = (grid_x + xy_pred[..., 0:1]) / W
        box_y = (grid_y + xy_pred[..., 1:2]) / H
        
        # Apply anchor scaling
        anchor_w = anchors[..., 2:3]  # [A, H, W, 1]
        anchor_h = anchors[..., 3:4]  # [A, H, W, 1]
        
        box_w = anchor_w * torch.exp(wh_pred[..., 0:1])
        box_h = anchor_h * torch.exp(wh_pred[..., 1:2])
        
        # Convert to corner coordinates for NMS
        x1 = box_x - box_w / 2
        y1 = box_y - box_h / 2
        x2 = box_x + box_w / 2
        y2 = box_y + box_h / 2
        
        # Stack boxes
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, A, H, W, 4]
        
        # Combine objectness and class scores
        scores = obj_pred * cls_pred  # [B, A, H, W, num_classes]
        
        # Get class indices
        class_scores, class_indices = torch.max(scores, dim=-1)  # [B, A, H, W]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'class_scores': class_scores,
            'class_indices': class_indices,
            'objectness': obj_pred,
            'raw_predictions': predictions
        }


class YOLOLoss(nn.Module):
    """
    YOLO loss function implementation.
    
    Combines:
    - Localization loss (bounding box coordinates)
    - Confidence loss (objectness scores)
    - Classification loss (class probabilities)
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        anchors: List[List[Tuple[int, int]]] = None,
        image_size: int = 416,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        # Default anchors if not provided
        if anchors is None:
            anchors = [
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)]
            ]
        
        self.anchors = anchors
        self.num_scales = len(anchors)
        
        # Loss components
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute Intersection over Union between boxes.
        
        Args:
            box1: Boxes in format [..., 4] (x1, y1, x2, y2)
            box2: Boxes in format [..., 4] (x1, y1, x2, y2)
            
        Returns:
            IoU values [...]
        """
        # Get intersection coordinates
        inter_x1 = torch.max(box1[..., 0], box2[..., 0])
        inter_y1 = torch.max(box1[..., 1], box2[..., 1])
        inter_x2 = torch.min(box1[..., 2], box2[..., 2])
        inter_y2 = torch.min(box1[..., 3], box2[..., 3])
        
        # Intersection area
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height
        
        # Union area
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = box1_area + box2_area - inter_area + 1e-6
        
        # IoU
        iou = inter_area / union_area
        
        return iou
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute YOLO loss.
        
        Args:
            predictions: Dictionary with predictions for each scale
            targets: List of target tensors for each scale
            
        Returns:
            Dictionary with loss components
        """
        total_loss = 0.0
        loss_dict = {
            'coord_loss': 0.0,
            'obj_loss': 0.0,
            'noobj_loss': 0.0,
            'cls_loss': 0.0,
        }
        
        # Process each scale
        for scale_idx in range(self.num_scales):
            scale_key = f'scale_{scale_idx}'
            
            if scale_key not in predictions:
                continue
                
            pred = predictions[scale_key]  # [B, A, H, W, 5+C]
            target = targets[scale_idx]    # [B, A, H, W, 5+C]
            
            B, A, H, W, _ = pred.shape
            
            # Create masks
            obj_mask = target[..., 4] > 0.5  # Objects
            noobj_mask = target[..., 4] < 0.5  # No objects
            
            num_objects = obj_mask.sum().item()
            if num_objects == 0:
                continue
            
            # === Coordinate Loss ===
            if obj_mask.any():
                # Extract predicted and target boxes
                pred_boxes = pred[obj_mask][:, :4]
                target_boxes = target[obj_mask][:, :4]
                
                # MSE loss for coordinates
                coord_loss = self.mse_loss(pred_boxes, target_boxes)
                loss_dict['coord_loss'] += coord_loss.item()
                total_loss += self.lambda_coord * coord_loss / num_objects
            
            # === Objectness Loss ===
            # Objects
            if obj_mask.any():
                pred_obj = pred[obj_mask][:, 4:5]
                target_obj = target[obj_mask][:, 4:5]
                
                obj_loss = F.binary_cross_entropy_with_logits(
                    pred_obj, target_obj, reduction='sum'
                )
                loss_dict['obj_loss'] += obj_loss.item()
                total_loss += self.lambda_obj * obj_loss / num_objects
            
            # No objects
            if noobj_mask.any():
                pred_noobj = pred[noobj_mask][:, 4:5]
                target_noobj = target[noobj_mask][:, 4:5]
                
                noobj_loss = F.binary_cross_entropy_with_logits(
                    pred_noobj, target_noobj, reduction='sum'
                )
                loss_dict['noobj_loss'] += noobj_loss.item()
                total_loss += self.lambda_noobj * noobj_loss / num_objects
            
            # === Classification Loss ===
            if obj_mask.any():
                pred_cls = pred[obj_mask][:, 5:]
                target_cls = target[obj_mask][:, 5:]
                
                cls_loss = F.binary_cross_entropy_with_logits(
                    pred_cls, target_cls, reduction='sum'
                )
                loss_dict['cls_loss'] += cls_loss.item()
                total_loss += self.lambda_cls * cls_loss / num_objects
        
        # Add total loss
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class YOLODetectionHead(nn.Module):
    """
    Complete YOLO detection head for multi-scale detection.
    
    Handles prediction generation, decoding, and post-processing.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int = 80,
        anchors: List[List[Tuple[int, int]]] = None,
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.num_classes = num_classes
        self.num_scales = len(in_channels_list)
        
        # Anchor generator
        self.anchor_generator = YOLOAnchorGenerator(anchors)
        self.num_anchors = self.anchor_generator.get_num_anchors()
        
        # Prediction heads for each scale
        self.pred_heads = nn.ModuleList([
            YOLOPredictionHead(
                in_channels=in_channels_list[i],
                num_classes=num_classes,
                num_anchors=self.num_anchors,
                use_mhc=use_mhc
            )
            for i in range(self.num_scales)
        ])
        
        # Decoder
        self.decoder = YOLODecoder(image_size=416)
        
        # Loss function
        self.loss_fn = YOLOLoss(
            num_classes=num_classes,
            anchors=anchors
        )
        
        # Grid sizes for different scales
        self.grid_sizes = [(13, 13), (26, 26), (52, 52)]
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[torch.Tensor]] = None,
        compute_loss: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through YOLO detection head.
        
        Args:
            features: Dictionary with features for each scale
            targets: Optional targets for loss computation
            compute_loss: Whether to compute loss
            
        Returns:
            Dictionary with predictions and optionally loss
        """
        predictions = {}
        decoded_outputs = {}
        
        # Process each scale
        for scale_idx in range(self.num_scales):
            scale_key = f'scale_{scale_idx}'
            feature_key = ['scale_small', 'scale_medium', 'scale_large'][scale_idx]
            
            if feature_key not in features:
                continue
            
            # Get features for this scale
            x = features[feature_key]
            
            # Get predictions
            pred = self.pred_heads[scale_idx](x)  # [B, A, H, W, 5+C]
            predictions[scale_key] = pred
            
            # Decode predictions
            anchors = self.anchor_generator(scale_idx)
            grid_size = self.grid_sizes[scale_idx]
            
            decoded = self.decoder(pred, anchors, grid_size)
            decoded_outputs[scale_key] = decoded
        
        # Compute loss if requested
        if compute_loss and targets is not None:
            loss_dict = self.loss_fn(predictions, targets)
            return {
                'predictions': predictions,
                'decoded': decoded_outputs,
                'loss': loss_dict
            }
        
        return {
            'predictions': predictions,
            'decoded': decoded_outputs
        }
    
    def post_process(
        self,
        decoded_outputs: Dict[str, Dict[str, torch.Tensor]],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_detections: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process detections with NMS.
        
        Args:
            decoded_outputs: Decoded predictions from forward()
            confidence_threshold: Minimum confidence score
            iou_threshold: NMS IoU threshold
            max_detections: Maximum number of detections per image
            
        Returns:
            List of detections per image
        """
        batch_detections = []
        
        for scale_key, scale_output in decoded_outputs.items():
            boxes = scale_output['boxes']  # [B, A, H, W, 4]
            scores = scale_output['class_scores']  # [B, A, H, W]
            class_indices = scale_output['class_indices']  # [B, A, H, W]
            
            B, A, H, W = scores.shape
            
            # Flatten predictions
            boxes_flat = boxes.reshape(B, -1, 4)  # [B, N, 4]
            scores_flat = scores.reshape(B, -1)    # [B, N]
            class_flat = class_indices.reshape(B, -1)  # [B, N]
            
            # Apply confidence threshold
            conf_mask = scores_flat > confidence_threshold
            
            detections = []
            for b in range(B):
                # Get indices for this batch
                mask = conf_mask[b]
                if not mask.any():
                    detections.append({
                        'boxes': torch.tensor([], device=boxes.device),
                        'scores': torch.tensor([], device=scores.device),
                        'labels': torch.tensor([], device=class_indices.device, dtype=torch.long)
                    })
                    continue
                
                # Filter by confidence
                batch_boxes = boxes_flat[b][mask]
                batch_scores = scores_flat[b][mask]
                batch_classes = class_flat[b][mask]
                
                # Apply NMS
                keep = self.non_max_suppression(
                    batch_boxes, batch_scores,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections
                )
                
                detections.append({
                    'boxes': batch_boxes[keep],
                    'scores': batch_scores[keep],
                    'labels': batch_classes[keep]
                })
            
            batch_detections.append(detections)
        
        # Combine detections from all scales
        combined_detections = []
        for b in range(B):
            boxes_list = []
            scores_list = []
            labels_list = []
            
            for detections in batch_detections:
                boxes_list.append(detections[b]['boxes'])
                scores_list.append(detections[b]['scores'])
                labels_list.append(detections[b]['labels'])
            
            # Concatenate
            all_boxes = torch.cat(boxes_list, dim=0)
            all_scores = torch.cat(scores_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            # Final NMS across scales
            if len(all_boxes) > 0:
                keep = self.non_max_suppression(
                    all_boxes, all_scores,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections
                )
                
                combined_detections.append({
                    'boxes': all_boxes[keep],
                    'scores': all_scores[keep],
                    'labels': all_labels[keep]
                })
            else:
                combined_detections.append({
                    'boxes': torch.tensor([], device=all_boxes.device),
                    'scores': torch.tensor([], device=all_scores.device),
                    'labels': torch.tensor([], device=all_labels.device, dtype=torch.long)
                })
        
        return combined_detections
    
    def non_max_suppression(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.5,
        max_detections: int = 100
    ) -> torch.Tensor:
        """
        Non-maximum suppression.
        
        Args:
            boxes: Boxes in format [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for suppression
            max_detections: Maximum number of detections to keep
            
        Returns:
            Indices of boxes to keep
        """
        if boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Sort by score (descending)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        boxes_sorted = boxes[sorted_indices]
        
        keep = []
        while len(sorted_indices) > 0:
            # Take highest score box
            current_idx = sorted_indices[0]
            keep.append(current_idx.item())
            
            if len(keep) >= max_detections:
                break
            
            # Remove current box
            sorted_indices = sorted_indices[1:]
            boxes_sorted = boxes_sorted[1:]
            
            if len(sorted_indices) == 0:
                break
            
            # Compute IoU with remaining boxes
            current_box = boxes[current_idx].unsqueeze(0)
            remaining_boxes = boxes[sorted_indices]
            
            iou = self.compute_iou(current_box, remaining_boxes)
            
            # Keep boxes with IoU < threshold
            keep_mask = iou < iou_threshold
            sorted_indices = sorted_indices[keep_mask.squeeze()]
            boxes_sorted = boxes_sorted[keep_mask.squeeze()]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes."""
        # Get intersection coordinates
        inter_x1 = torch.max(box1[..., 0], box2[..., 0])
        inter_y1 = torch.max(box1[..., 1], box2[..., 1])
        inter_x2 = torch.min(box1[..., 2], box2[..., 2])
        inter_y2 = torch.min(box1[..., 3], box2[..., 3])
        
        # Intersection area
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height
        
        # Box areas
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        
        # Union area
        union_area = box1_area + box2_area - inter_area + 1e-6
        
        # IoU
        iou = inter_area / union_area
        
        return iou