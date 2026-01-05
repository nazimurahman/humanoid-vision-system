# src/models/vision_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from .manifold_layers import ManifoldHyperConnection, RMSNorm


class ConvMHCLayer(nn.Module):
    """
    Convolutional layer with Manifold Hyper-Connections.
    
    Combines standard convolution with mHC stability mechanisms.
    Designed for efficient robotic vision processing.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        expansion_rate: int = 4,
        use_mhc: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_mhc = use_mhc
        
        # Auto padding for same output size
        if padding is None:
            padding = kernel_size // 2
        
        # Standard convolutional components
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Activation function
        if activation == 'silu':
            self.activation = nn.SiLU()  # Swish activation (efficient, smooth)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Manifold Hyper-Connection for feature enhancement
        if use_mhc:
            self.mhc = ManifoldHyperConnection(
                input_dim=out_channels,
                expansion_rate=expansion_rate
            )
        else:
            self.mhc = None
        
        # Residual connection conditions
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        
        # Channel attention (optional)
        if use_mhc and out_channels >= 32:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, 1),
                self.activation,
                nn.Conv2d(out_channels // 4, out_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.channel_attention = None
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
        if self.bn.weight is not None:
            nn.init.ones_(self.bn.weight)
        if self.bn.bias is not None:
            nn.init.zeros_(self.bn.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional MHC enhancement.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C_out, H_out, W_out]
        """
        identity = x
        
        # Standard convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        # Apply MHC at channel level if enabled
        if self.mhc is not None:
            B, C, H, W = x.shape
            
            # Reshape for MHC: treat spatial positions as batch
            x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            x_transformed = self.mhc(x_reshaped)
            x = x_transformed.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # Apply channel attention if available
            if self.channel_attention is not None:
                attention = self.channel_attention(x)
                x = x * attention
        
        # Residual connection (if dimensions match)
        if self.use_residual:
            x = x + identity
        
        return x


class ResidualMHCLayer(nn.Module):
    """
    Residual block with multiple ConvMHCLayers.
    
    Standard pattern: Conv → Conv + Residual
    with MHC stabilization throughout.
    """
    
    def __init__(
        self,
        channels: int,
        num_blocks: int = 2,
        expansion_rate: int = 4,
        bottleneck: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.num_blocks = num_blocks
        
        layers = []
        
        # First block (optional bottleneck)
        if bottleneck and channels >= 64:
            layers.append(ConvMHCLayer(
                channels, channels // 2,
                kernel_size=1,
                expansion_rate=expansion_rate
            ))
            layers.append(ConvMHCLayer(
                channels // 2, channels,
                kernel_size=3,
                expansion_rate=expansion_rate
            ))
        else:
            for i in range(num_blocks):
                layers.append(ConvMHCLayer(
                    channels, channels,
                    kernel_size=3,
                    expansion_rate=expansion_rate
                ))
        
        self.blocks = nn.Sequential(*layers)
        
        # Final projection if needed
        if bottleneck and channels >= 64:
            self.projection = ConvMHCLayer(
                channels, channels,
                kernel_size=1,
                expansion_rate=expansion_rate
            )
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.blocks(x)
        x = self.projection(x)
        x = x + identity  # Residual connection
        return x


class HybridVisionBackbone(nn.Module):
    """
    Hybrid Vision Backbone for robotic perception.
    
    Combines:
    1. Efficient CNN for local feature extraction
    2. MHC layers for training stability
    3. Multi-scale outputs for object detection
    
    Designed for:
    - Real-time inference (<50ms)
    - Low GPU memory (<4GB)
    - Edge deployment (Jetson, Xavier)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 32,
        num_blocks: List[int] = [2, 3, 4, 2],
        use_mhc: bool = True,
        activation: str = 'silu',
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.use_mhc = use_mhc
        self.dropout_rate = dropout_rate
        
        # === Initial Stem ===
        # High-resolution processing with minimal downsampling
        self.stem = nn.Sequential(
            ConvMHCLayer(
                input_channels, base_channels,
                kernel_size=3, stride=2, padding=1,
                use_mhc=use_mhc, activation=activation
            ),
            ConvMHCLayer(
                base_channels, base_channels,
                kernel_size=3, stride=1, padding=1,
                use_mhc=use_mhc, activation=activation
            ),
            ConvMHCLayer(
                base_channels, base_channels * 2,
                kernel_size=3, stride=1, padding=1,
                use_mhc=use_mhc, activation=activation
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # === Multi-scale Feature Extraction ===
        self.stages = nn.ModuleList()
        current_channels = base_channels * 2
        
        # Stage configurations
        stage_channels = [
            current_channels,      # Stage 1
            current_channels * 2,  # Stage 2  
            current_channels * 4,  # Stage 3
            current_channels * 8   # Stage 4
        ]
        
        for i, (num_layers, out_channels) in enumerate(zip(num_blocks, stage_channels)):
            # First layer may need downsampling
            stride = 2 if i > 0 else 1
            
            stage_layers = []
            
            # First layer (potentially with channel change)
            stage_layers.append(ConvMHCLayer(
                current_channels, out_channels,
                kernel_size=3, stride=stride,
                use_mhc=use_mhc, activation=activation
            ))
            
            # Additional residual blocks
            for _ in range(1, num_layers):
                stage_layers.append(ResidualMHCLayer(
                    out_channels,
                    num_blocks=2,
                    expansion_rate=4,
                    bottleneck=True
                ))
            
            stage = nn.Sequential(*stage_layers)
            self.stages.append(stage)
            current_channels = out_channels
        
        # === Feature Enhancement ===
        # MHC-based feature refinement
        self.enhance_large = ManifoldHyperConnection(
            input_dim=stage_channels[-1],
            expansion_rate=4
        ) if use_mhc else nn.Identity()
        
        self.enhance_medium = ManifoldHyperConnection(
            input_dim=stage_channels[-2],
            expansion_rate=4
        ) if use_mhc else nn.Identity()
        
        self.enhance_small = ManifoldHyperConnection(
            input_dim=stage_channels[-3],
            expansion_rate=4
        ) if use_mhc else nn.Identity()
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Output channel specifications
        self.output_channels = {
            'stem': base_channels * 2,
            'stage_1': stage_channels[0],
            'stage_2': stage_channels[1],  # Small objects
            'stage_3': stage_channels[2],  # Medium objects
            'stage_4': stage_channels[3],  # Large objects
        }
        
        # Spatial reduction factors
        self.stride_factors = {
            'stem': 4,      # 2 (stem) * 2 (pool)
            'stage_1': 4,   # Same as stem
            'stage_2': 8,   # Additional 2x downsampling
            'stage_3': 16,  # Additional 2x downsampling  
            'stage_4': 32   # Additional 2x downsampling
        }
        
        print(f"Backbone initialized with channels: {self.output_channels}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from input image.
        
        Args:
            x: Input tensor [B, 3, H, W] (H, W typically 416x416)
            
        Returns:
            Dictionary of features at different scales:
            - 'scale_small': High resolution, shallow features [B, C1, H/8, W/8]
            - 'scale_medium': Medium resolution [B, C2, H/16, W/16]
            - 'scale_large': Low resolution, deep features [B, C3, H/32, W/32]
        """
        features = {}
        
        # === Stem ===
        x = self.stem(x)
        features['stem'] = x  # [B, 64, H/4, W/4]
        
        # === Stage 1 ===
        x = self.stages[0](x)
        features['stage_1'] = x  # [B, 64, H/4, W/4]
        
        # === Stage 2 (Small objects) ===
        x = self.stages[1](x)
        features['stage_2'] = x  # [B, 128, H/8, W/8]
        
        # === Stage 3 (Medium objects) ===
        x = self.stages[2](x)
        features['stage_3'] = x  # [B, 256, H/16, W/16]
        
        # === Stage 4 (Large objects) ===
        x = self.stages[3](x)
        features['stage_4'] = x  # [B, 512, H/32, W/32]
        
        # === Feature Enhancement with MHC ===
        enhanced_features = {}
        
        # Small scale (high resolution)
        small_feat = features['stage_2']
        if self.use_mhc:
            B, C, H, W = small_feat.shape
            small_flat = small_feat.permute(0, 2, 3, 1).reshape(-1, C)
            small_enhanced = self.enhance_small(small_flat)
            small_feat = small_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)
        enhanced_features['scale_small'] = self.dropout(small_feat)
        
        # Medium scale
        medium_feat = features['stage_3']
        if self.use_mhc:
            B, C, H, W = medium_feat.shape
            medium_flat = medium_feat.permute(0, 2, 3, 1).reshape(-1, C)
            medium_enhanced = self.enhance_medium(medium_flat)
            medium_feat = medium_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)
        enhanced_features['scale_medium'] = self.dropout(medium_feat)
        
        # Large scale (deep features)
        large_feat = features['stage_4']
        if self.use_mhc:
            B, C, H, W = large_feat.shape
            large_flat = large_feat.permute(0, 2, 3, 1).reshape(-1, C)
            large_enhanced = self.enhance_large(large_flat)
            large_feat = large_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)
        enhanced_features['scale_large'] = self.dropout(large_feat)
        
        # Add raw features for reference
        enhanced_features['raw_features'] = features
        
        return enhanced_features
    
    def get_output_channels(self) -> Dict[str, int]:
        """Get output channels for each scale."""
        return {
            'scale_small': self.output_channels['stage_2'],
            'scale_medium': self.output_channels['stage_3'],
            'scale_large': self.output_channels['stage_4']
        }
    
    def get_stride_factors(self) -> Dict[str, int]:
        """Get spatial reduction factors for each scale."""
        return {
            'scale_small': self.stride_factors['stage_2'],
            'scale_medium': self.stride_factors['stage_3'],
            'scale_large': self.stride_factors['stage_4']
        }
    
    def compute_flops(self, input_size: Tuple[int, int] = (416, 416)) -> Dict[str, float]:
        """
        Compute FLOPs for each stage (approximate).
        
        Args:
            input_size: (height, width) of input image
            
        Returns:
            Dictionary of FLOPs in GFLOPs
        """
        import numpy as np
        
        H, W = input_size
        flops = {}
        
        # Helper function to compute conv FLOPs
        def conv_flops(C_in, C_out, K, H_out, W_out, groups=1):
            flops_per_instance = K * K * C_in // groups
            total_instances = H_out * W_out * C_out
            return flops_per_instance * total_instances
        
        # Stem FLOPs
        # First conv: 3→32, 3x3, stride 2
        H1, W1 = H // 2, W // 2
        flops['stem_conv1'] = conv_flops(3, 32, 3, H1, W1)
        
        # Second conv: 32→32
        flops['stem_conv2'] = conv_flops(32, 32, 3, H1, W1)
        
        # Third conv: 32→64
        flops['stem_conv3'] = conv_flops(32, 64, 3, H1, W1)
        
        # Max pool
        H2, W2 = H1 // 2, W1 // 2
        flops['stem_pool'] = H2 * W2 * 64 * 4  # 2x2 max pool
        
        # Total stem FLOPs
        flops['stem_total'] = sum([flops[k] for k in ['stem_conv1', 'stem_conv2', 'stem_conv3', 'stem_pool']])
        
        # Convert to GFLOPs
        for key in list(flops.keys()):
            flops[key] = flops[key] / 1e9
        
        return flops