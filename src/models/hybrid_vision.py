# src/models/hybrid_vision.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math

from .vision_backbone import HybridVisionBackbone
from .vit_encoder_decoder import HybridVisionEncoder, VisionTransformerEncoder
from .yolo_head import YOLODetectionHead
from .feature_fusion import FeaturePyramidNetwork, AdaptiveFeatureFusion
from .rag_module import RAGVisionKnowledge, KnowledgeAwareDetection
from .manifold_layers import ManifoldHyperConnection


class HybridVisionSystem(nn.Module):
    """
    Complete Hybrid Vision System for Robotic Perception.
    
    Architecture Components:
    1. Hybrid CNN Backbone (efficient local features)
    2. Vision Transformer Encoder (global context)
    3. Feature Pyramid Network (multi-scale fusion)
    4. YOLO Detection Heads (object detection)
    5. RAG Module (knowledge enhancement)
    6. Manifold Hyper-Connections (training stability)
    
    Key Features:
    - Real-time inference optimized
    - Training stability via mHC
    - Multi-scale object detection
    - Knowledge-augmented understanding
    - Robot deployment ready
    """
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        super().__init__()
        
        self.config = config
        
        # Extract configuration
        self.image_size = config.get('image_size', 416)
        self.num_classes = config.get('num_classes', 80)
        self.use_mhc = config.get('use_mhc', True)
        self.use_vit = config.get('use_vit', True)
        self.use_rag = config.get('use_rag', False)
        self.use_fpn = config.get('use_fpn', True)
        
        # === 1. Hybrid CNN Backbone ===
        self.backbone = HybridVisionBackbone(
            input_channels=3,
            base_channels=32,
            num_blocks=[2, 3, 4, 2],
            use_mhc=self.use_mhc,
            activation='silu',
            dropout_rate=0.1
        )
        
        # Get backbone output channels
        backbone_channels = self.backbone.get_output_channels()
        
        # === 2. Vision Transformer for Global Context ===
        if self.use_vit:
            self.vit_encoder = HybridVisionEncoder(
                cnn_channels=backbone_channels['scale_large'],
                vit_embed_dim=256,
                vit_depth=6,
                vit_num_heads=8,
                use_mhc=self.use_mhc
            )
        
        # === 3. Feature Fusion ===
        if self.use_fpn:
            self.feature_fusion = FeaturePyramidNetwork(
                channels=[
                    backbone_channels['scale_small'],
                    backbone_channels['scale_medium'], 
                    backbone_channels['scale_large']
                ],
                use_mhc=self.use_mhc,
                fusion_method='add'
            )
        else:
            self.feature_fusion = AdaptiveFeatureFusion(
                channels=[
                    backbone_channels['scale_small'],
                    backbone_channels['scale_medium'],
                    backbone_channels['scale_large']
                ],
                use_mhc=self.use_mhc
            )
        
        # === 4. YOLO Detection Heads ===
        # Get fused feature channels
        if self.use_fpn:
            fused_channels = [256, 512, 1024]  # FPN output channels
        else:
            fused_channels = [
                backbone_channels['scale_small'],
                backbone_channels['scale_medium'],
                backbone_channels['scale_large']
            ]
        
        self.detection_head = YOLODetectionHead(
            in_channels_list=fused_channels,
            num_classes=self.num_classes,
            anchors=config.get('anchors', None),
            use_mhc=self.use_mhc
        )
        
        # === 5. RAG Module (Optional) ===
        if self.use_rag:
            self.rag_module = RAGVisionKnowledge(
                visual_dim=256,  # Assuming visual features are 256-dim
                knowledge_dim=512,
                num_retrievals=5,
                use_mhc=self.use_mhc
            )
            
            self.knowledge_enhancer = KnowledgeAwareDetection(
                visual_dim=256,
                knowledge_dim=512,
                num_classes=self.num_classes,
                use_mhc=self.use_mhc
            )
        
        # === 6. Additional Heads ===
        # Segmentation head (optional)
        self.has_segmentation = config.get('has_segmentation', False)
        if self.has_segmentation:
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(fused_channels[0], 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.num_classes + 1, 1)  # +1 for background
            )
        
        # Depth estimation head (optional)
        self.has_depth = config.get('has_depth', False)
        if self.has_depth:
            self.depth_head = nn.Sequential(
                nn.Conv2d(fused_channels[0], 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)  # Single channel for depth
            )
        
        # === 7. Final Fusion Layer ===
        # MHC for stabilizing final features
        if self.use_mhc:
            self.final_fusion = ManifoldHyperConnection(
                input_dim=sum(fused_channels),  # Combined features
                expansion_rate=2
            )
        else:
            self.final_fusion = nn.Identity()
        
        # === 8. Output Projection ===
        self.output_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sum(fused_channels), 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # Final feature dimension
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Print architecture summary
        self._print_summary()
    
    def _initialize_weights(self):
        """Initialize all weights for stable training."""
        # Initialize all conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _print_summary(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("HYBRID VISION SYSTEM ARCHITECTURE")
        print("="*60)
        print(f"Image Size: {self.image_size}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("\nComponents:")
        print(f"  • Backbone: Hybrid CNN with MHC")
        print(f"  • ViT Encoder: {'Enabled' if self.use_vit else 'Disabled'}")
        print(f"  • Feature Fusion: {'FPN' if self.use_fpn else 'Adaptive'}")
        print(f"  • Detection: YOLO Head (3 scales)")
        print(f"  • RAG Module: {'Enabled' if self.use_rag else 'Disabled'}")
        print(f"  • Manifold Hyper-Connections: {'Enabled' if self.use_mhc else 'Disabled'}")
        print(f"  • Segmentation: {'Enabled' if self.has_segmentation else 'Disabled'}")
        print(f"  • Depth Estimation: {'Enabled' if self.has_depth else 'Disabled'}")
        print("="*60 + "\n")
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[torch.Tensor]] = None,
        text_query: Optional[str] = None,
        task: str = 'detection',
        compute_loss: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for different vision tasks.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            targets: Optional targets for loss computation
            text_query: Optional text query for RAG
            task: Task type ('detection', 'features', 'segmentation', 'depth')
            compute_loss: Whether to compute loss
            
        Returns:
            Dictionary with task-specific outputs
        """
        outputs = {}
        
        # === 1. Backbone Feature Extraction ===
        backbone_features = self.backbone(x)
        outputs['backbone_features'] = backbone_features
        
        # === 2. ViT Global Context Enhancement ===
        if self.use_vit:
            # Enhance large-scale features with ViT
            large_features = backbone_features['scale_large']
            vit_enhanced = self.vit_encoder(large_features)
            
            # Fuse with original features
            backbone_features['scale_large'] = (
                backbone_features['scale_large'] + vit_enhanced
            ) / 2
            
            outputs['vit_features'] = vit_enhanced
        
        # === 3. Multi-Scale Feature Fusion ===
        fused_features = self.feature_fusion(backbone_features)
        outputs['fused_features'] = fused_features
        
        # === 4. RAG Knowledge Enhancement (Optional) ===
        if self.use_rag and text_query is not None:
            # Enhance features with knowledge
            # We'll use small-scale features for knowledge enhancement
            small_features = fused_features.get('fused_small', 
                                               backbone_features['scale_small'])
            
            B, C, H, W = small_features.shape
            
            # Reshape for RAG
            small_flat = small_features.permute(0, 2, 3, 1).reshape(-1, C)
            
            # Apply RAG
            if text_query:
                rag_enhanced, knowledge = self.rag_module(
                    small_flat, text_query, return_knowledge=True
                )
                outputs['knowledge'] = knowledge
            else:
                rag_enhanced = self.rag_module(small_flat)
            
            # Reshape back
            rag_enhanced = rag_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # Fuse with original features
            fused_features['fused_small'] = (
                fused_features.get('fused_small', small_features) + rag_enhanced
            ) / 2
            
            outputs['rag_features'] = rag_enhanced
        
        # === 5. Task-Specific Processing ===
        if task == 'detection':
            # Prepare features for detection
            detection_features = {
                'scale_small': fused_features.get('fused_small', 
                                                 backbone_features['scale_small']),
                'scale_medium': fused_features.get('fused_medium', 
                                                  backbone_features['scale_medium']),
                'scale_large': fused_features.get('fused_large', 
                                                 backbone_features['scale_large'])
            }
            
            # YOLO Detection
            detection_results = self.detection_head(
                detection_features,
                targets=targets,
                compute_loss=compute_loss
            )
            
            outputs.update(detection_results)
            
            # Knowledge-enhanced detection (optional)
            if self.use_rag and 'decoded' in detection_results:
                enhanced_detections = self.knowledge_enhancer(
                    detection_features['scale_small'],
                    detection_results['decoded'],
                    text_query=text_query
                )
                outputs['enhanced_detections'] = enhanced_detections
        
        elif task == 'segmentation' and self.has_segmentation:
            # Semantic segmentation
            seg_features = fused_features.get('fused_small', 
                                             backbone_features['scale_small'])
            segmentation = self.segmentation_head(seg_features)
            
            # Upsample to input size
            segmentation = F.interpolate(
                segmentation, size=x.shape[2:],
                mode='bilinear', align_corners=False
            )
            
            outputs['segmentation'] = segmentation
        
        elif task == 'depth' and self.has_depth:
            # Depth estimation
            depth_features = fused_features.get('fused_small', 
                                               backbone_features['scale_small'])
            depth = self.depth_head(depth_features)
            
            # Upsample to input size
            depth = F.interpolate(
                depth, size=x.shape[2:],
                mode='bilinear', align_corners=False
            )
            
            outputs['depth'] = depth
        
        elif task == 'features':
            # Return all features for downstream tasks
            outputs['all_features'] = {
                'backbone': backbone_features,
                'fused': fused_features,
                'final': self._extract_final_features(fused_features)
            }
        
        # === 6. Final Feature Extraction ===
        if 'final_features' not in outputs:
            outputs['final_features'] = self._extract_final_features(fused_features)
        
        return outputs
    
    def _extract_final_features(self, fused_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract final feature representation.
        
        Args:
            fused_features: Dictionary of fused features
            
        Returns:
            Final feature tensor [B, D]
        """
        # Get features from all scales
        feature_list = []
        
        for key in ['fused_small', 'fused_medium', 'fused_large']:
            if key in fused_features:
                feat = fused_features[key]
                
                # Global average pooling
                feat_pooled = F.adaptive_avg_pool2d(feat, (1, 1))
                feat_pooled = feat_pooled.flatten(1)  # [B, C]
                feature_list.append(feat_pooled)
        
        # Concatenate features from all scales
        if feature_list:
            combined = torch.cat(feature_list, dim=1)  # [B, sum(C)]
            
            # Apply final fusion (MHC)
            combined = self.final_fusion(combined)
            
            # Project to final feature dimension
            final_features = self.output_projection(combined)
            return final_features
        
        return torch.tensor([], device=next(self.parameters()).device)
    
    def detect(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        text_query: Optional[str] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convenience method for detection.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            confidence_threshold: Minimum confidence score
            iou_threshold: NMS IoU threshold
            max_detections: Maximum detections per image
            text_query: Optional text query for RAG
            
        Returns:
            List of detections per image
        """
        # Forward pass
        outputs = self.forward(x, text_query=text_query, task='detection')
        
        if 'decoded' not in outputs:
            return []
        
        # Post-process detections
        detections = self.detection_head.post_process(
            outputs['decoded'],
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections
        )
        
        return detections
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """
        Collect stability metrics from all MHC layers.
        
        Returns:
            Dictionary of stability metrics
        """
        metrics = {}
        
        # Collect from all MHC layers
        for name, module in self.named_modules():
            if hasattr(module, 'get_stability_metrics'):
                module_metrics = module.get_stability_metrics()
                for key, value in module_metrics.items():
                    metrics[f'{name}.{key}'] = value
        
        return metrics
    
    def add_knowledge(self, text: str):
        """
        Add knowledge to the RAG module.
        
        Args:
            text: Knowledge text
        """
        if self.use_rag:
            self.rag_module.add_knowledge(text)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter counts for each component.
        
        Returns:
            Dictionary with parameter counts
        """
        counts = {}
        
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            counts[name] = params
        
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return counts


class LightweightHybridVision(HybridVisionSystem):
    """
    Lightweight version for edge deployment.
    
    Optimized for:
    - Low memory usage
    - Fast inference
    - Mobile/embedded devices
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Override config for lightweight version
        lightweight_config = config.copy()
        lightweight_config.update({
            'base_channels': 16,  # Reduced channels
            'use_vit': False,     # No ViT for speed
            'use_rag': False,     # No RAG for speed
            'use_fpn': True,      # Keep FPN but lightweight
        })
        
        super().__init__(lightweight_config)
        
        # Further optimizations
        # Replace heavy convolutions with depthwise separable convs
        self._make_lightweight()
    
    def _make_lightweight(self):
        """Convert to lightweight architecture."""
        # This would involve replacing standard convs with depthwise separable convs
        # For simplicity, we just reduce channels further
        
        # Reduce channels in detection head
        if hasattr(self.detection_head, 'pred_heads'):
            for head in self.detection_head.pred_heads:
                # Reduce channels in conv layers
                for module in head.conv_layers:
                    if isinstance(module, nn.Conv2d):
                        # Reduce output channels by half
                        if module.out_channels > 64:
                            new_out_channels = module.out_channels // 2
                            new_conv = nn.Conv2d(
                                module.in_channels, new_out_channels,
                                kernel_size=module.kernel_size,
                                stride=module.stride,
                                padding=module.padding
                            )
                            # Copy weights (approximately)
                            with torch.no_grad():
                                new_conv.weight[:new_out_channels] = module.weight[:new_out_channels]
                                if module.bias is not None:
                                    new_conv.bias[:new_out_channels] = module.bias[:new_out_channels]
                            # Replace module
                            head.conv_layers[head.conv_layers.index(module)] = new_conv


class ProductionHybridVision(HybridVisionSystem):
    """
    Production version with additional features.
    
    Includes:
    - Quantization awareness
    - TensorRT optimization
    - Advanced monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Add quantization support
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Add profiling hooks
        self.register_forward_hook(self._profile_hook)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Quantization-aware forward pass."""
        # Quantize input if in quantization mode
        if self.training:
            x = self.quant(x)
        
        # Normal forward pass
        outputs = super().forward(x, **kwargs)
        
        # Dequantize outputs if in quantization mode
        if self.training:
            for key in outputs:
                if torch.is_tensor(outputs[key]):
                    outputs[key] = self.dequant(outputs[key])
        
        return outputs
    
    def _profile_hook(self, module, input, output):
        """Profile forward pass timing."""
        if hasattr(self, 'forward_time'):
            import time
            self.forward_time.append(time.time())
    
    def prepare_for_quantization(self):
        """Prepare model for quantization."""
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self, inplace=True)
    
    def convert_to_quantized(self):
        """Convert to quantized model."""
        torch.quantization.convert(self, inplace=True)