# src/models/feature_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from .manifold_layers import ManifoldHyperConnection


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    
    Combines features from different scales to create
    rich, multi-resolution representations.
    """
    
    def __init__(
        self,
        channels: List[int],
        use_mhc: bool = True,
        fusion_method: str = 'add'
    ):
        super().__init__()
        
        self.channels = channels  # [C1, C2, C3] from small to large
        self.num_scales = len(channels)
        self.fusion_method = fusion_method
        
        # Lateral connections (1x1 conv to match channels)
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_scales):
            lateral_conv = nn.Conv2d(
                channels[i], 256,  # Standardize to 256 channels
                kernel_size=1
            )
            self.lateral_convs.append(lateral_conv)
        
        # Feature refinement convs
        self.refinement_convs = nn.ModuleList()
        for i in range(self.num_scales):
            refinement_conv = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.refinement_convs.append(refinement_conv)
        
        # MHC enhancement for fusion
        if use_mhc:
            self.mhc_fusions = nn.ModuleList([
                ManifoldHyperConnection(input_dim=256, expansion_rate=2)
                for _ in range(self.num_scales)
            ])
        else:
            self.mhc_fusions = nn.ModuleList([nn.Identity() for _ in range(self.num_scales)])
        
        # Output projections
        self.output_convs = nn.ModuleList()
        output_channels = [256, 512, 1024]  # Different channels for different scales
        for i in range(self.num_scales):
            output_conv = nn.Conv2d(256, output_channels[i], kernel_size=1)
            self.output_convs.append(output_conv)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-scale features using FPN.
        
        Args:
            features: Dictionary with features at different scales
                - 'scale_small': [B, C1, H1, W1]
                - 'scale_medium': [B, C2, H2, W2] 
                - 'scale_large': [B, C3, H3, W3]
            
        Returns:
            Dictionary with fused features at each scale
        """
        # Extract features in order: large → medium → small
        f_large = features.get('scale_large')  # Deep, low-resolution
        f_medium = features.get('scale_medium') # Mid-resolution
        f_small = features.get('scale_small')   # Shallow, high-resolution
        
        # Apply lateral convolutions
        p_large = self.lateral_convs[2](f_large) if f_large is not None else None
        p_medium = self.lateral_convs[1](f_medium) if f_medium is not None else None
        p_small = self.lateral_convs[0](f_small) if f_small is not None else None
        
        # === Top-down pathway ===
        fused_features = {}
        
        # Start from largest scale
        if p_large is not None:
            # Refine large features
            p_large_refined = self.refinement_convs[2](p_large)
            p_large_refined = self.mhc_fusions[2](p_large_refined)
            fused_features['fused_large'] = self.output_convs[2](p_large_refined)
            
            # Upsample to medium resolution
            if p_medium is not None:
                p_large_up = F.interpolate(
                    p_large_refined, 
                    size=p_medium.shape[2:],
                    mode='nearest'
                )
                
                # Fuse with medium features
                if self.fusion_method == 'add':
                    p_medium_fused = p_medium + p_large_up
                else:  # concat
                    p_medium_fused = torch.cat([p_medium, p_large_up], dim=1)
                
                # Refine medium features
                p_medium_refined = self.refinement_convs[1](p_medium_fused)
                p_medium_refined = self.mhc_fusions[1](p_medium_refined)
                fused_features['fused_medium'] = self.output_convs[1](p_medium_refined)
                
                # Upsample to small resolution
                if p_small is not None:
                    p_medium_up = F.interpolate(
                        p_medium_refined,
                        size=p_small.shape[2:],
                        mode='nearest'
                    )
                    
                    # Fuse with small features
                    if self.fusion_method == 'add':
                        p_small_fused = p_small + p_medium_up
                    else:
                        p_small_fused = torch.cat([p_small, p_medium_up], dim=1)
                    
                    # Refine small features
                    p_small_refined = self.refinement_convs[0](p_small_fused)
                    p_small_refined = self.mhc_fusions[0](p_small_refined)
                    fused_features['fused_small'] = self.output_convs[0](p_small_refined)
        
        return fused_features


class MultiScaleFeatureFusion(nn.Module):
    """
    Advanced multi-scale feature fusion with attention.
    
    Uses attention mechanisms to selectively combine features
    from different scales based on their relevance.
    """
    
    def __init__(
        self,
        channels: List[int],
        use_mhc: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.use_mhc = use_mhc
        self.use_attention = use_attention
        
        # Feature projection to common dimension
        self.projections = nn.ModuleList()
        common_dim = 256
        for c in channels:
            projection = nn.Sequential(
                nn.Conv2d(c, common_dim, kernel_size=1),
                nn.BatchNorm2d(common_dim),
                nn.ReLU(inplace=True)
            )
            self.projections.append(projection)
        
        # Cross-scale attention
        if use_attention:
            self.attention = CrossScaleAttention(
                dim=common_dim,
                num_scales=len(channels)
            )
        else:
            self.attention = None
        
        # MHC-based fusion
        if use_mhc:
            self.mhc_fusion = ManifoldHyperConnection(
                input_dim=common_dim,
                expansion_rate=2
            )
        else:
            self.mhc_fusion = nn.Identity()
        
        # Output projections
        self.output_projections = nn.ModuleList()
        for i, c in enumerate(channels):
            output_proj = nn.Conv2d(common_dim, c, kernel_size=1)
            self.output_projections.append(output_proj)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fuse features with cross-scale attention.
        
        Args:
            features: Dictionary of features at different scales
            
        Returns:
            Fused features at each scale
        """
        # Extract and project features
        projected_features = []
        feature_keys = ['scale_small', 'scale_medium', 'scale_large']
        
        for i, key in enumerate(feature_keys):
            if key in features:
                feat = features[key]
                projected = self.projections[i](feat)
                projected_features.append(projected)
            else:
                projected_features.append(None)
        
        # Apply cross-scale attention if enabled
        if self.attention is not None and all(f is not None for f in projected_features):
            attended_features = self.attention(projected_features)
        else:
            attended_features = projected_features
        
        # MHC enhancement and output projection
        fused_features = {}
        for i, key in enumerate(feature_keys):
            if attended_features[i] is not None:
                # Apply MHC fusion
                B, C, H, W = attended_features[i].shape
                feat_flat = attended_features[i].permute(0, 2, 3, 1).reshape(-1, C)
                feat_fused = self.mhc_fusion(feat_flat)
                feat_fused = feat_fused.reshape(B, H, W, C).permute(0, 3, 1, 2)
                
                # Project to output dimension
                feat_out = self.output_projections[i](feat_fused)
                fused_features[key] = feat_out
        
        return fused_features


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention for feature fusion.
    
    Allows features at different scales to attend to each other,
    enabling adaptive fusion based on content.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query, key, value projections
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        
        # Output projection
        self.out_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        
        # Layer normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_scales)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale
        self.scale = self.head_dim ** -0.5
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-scale attention.
        
        Args:
            features: List of feature tensors at different scales
            
        Returns:
            List of attended features
        """
        B = features[0].shape[0]
        attended_features = []
        
        # Reshape features for attention
        reshaped_features = []
        for feat in features:
            B, C, H, W = feat.shape
            feat_reshaped = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            reshaped_features.append(feat_reshaped)
        
        # Apply cross-attention
        for i in range(self.num_scales):
            # Get queries for current scale
            q = self.q_projs[i](reshaped_features[i])  # [B, N_i, D]
            
            # Attend to all scales (including self)
            attention_outputs = []
            for j in range(self.num_scales):
                # Get keys and values for scale j
                k = self.k_projs[j](reshaped_features[j])  # [B, N_j, D]
                v = self.v_projs[j](reshaped_features[j])  # [B, N_j, D]
                
                # Compute attention
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Apply attention
                attn_output = torch.matmul(attn_weights, v)  # [B, N_i, D]
                attention_outputs.append(attn_output)
            
            # Combine attention outputs
            combined = sum(attention_outputs) / self.num_scales
            
            # Output projection and residual
            output = self.out_projs[i](combined)
            output = output + reshaped_features[i]  # Residual
            output = self.norms[i](output)
            
            # Reshape back to spatial format
            B, N, D = output.shape
            H = W = int(N ** 0.5)
            output = output.reshape(B, H, W, D).permute(0, 3, 1, 2)
            attended_features.append(output)
        
        return attended_features


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive feature fusion with learnable weights.
    
    Learns to combine features from different scales
    based on their importance for the task.
    """
    
    def __init__(
        self,
        channels: List[int],
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.num_scales = len(channels)
        
        # Weight generators
        self.weight_generators = nn.ModuleList()
        for c in channels:
            weight_gen = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 4, self.num_scales, 1),
                nn.Softmax(dim=1)
            )
            self.weight_generators.append(weight_gen)
        
        # MHC for feature enhancement
        if use_mhc:
            self.mhc_enhance = nn.ModuleList([
                ManifoldHyperConnection(input_dim=c, expansion_rate=2)
                for c in channels
            ])
        else:
            self.mhc_enhance = nn.ModuleList([nn.Identity() for _ in channels])
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Adaptively fuse features.
        
        Args:
            features: Dictionary of features at different scales
            
        Returns:
            Fused features
        """
        # Extract features
        feat_list = []
        for key in ['scale_small', 'scale_medium', 'scale_large']:
            if key in features:
                feat_list.append(features[key])
        
        # Generate weights for each scale
        weights_list = []
        for i, feat in enumerate(feat_list):
            weights = self.weight_generators[i](feat)  # [B, num_scales, 1, 1]
            weights_list.append(weights)
        
        # Fuse features
        fused_features = {}
        for i, key in enumerate(['scale_small', 'scale_medium', 'scale_large']):
            if i < len(feat_list):
                # Collect contributions from all scales
                contributions = []
                for j, feat in enumerate(feat_list):
                    # Resize if needed
                    if feat.shape[2:] != feat_list[i].shape[2:]:
                        feat_resized = F.interpolate(
                            feat, size=feat_list[i].shape[2:],
                            mode='bilinear', align_corners=False
                        )
                    else:
                        feat_resized = feat
                    
                    # Weight contribution
                    weight = weights_list[j][:, i:i+1]  # [B, 1, 1, 1]
                    contribution = feat_resized * weight
                    contributions.append(contribution)
                
                # Sum contributions
                fused = sum(contributions)
                
                # MHC enhancement
                B, C, H, W = fused.shape
                fused_flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
                fused_enhanced = self.mhc_enhance[i](fused_flat)
                fused = fused_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)
                
                fused_features[key] = fused
        
        return fused_features