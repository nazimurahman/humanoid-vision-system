# src/models/vit_encoder_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from .manifold_layers import ManifoldHyperConnection, MultiHeadManifoldAttention, RMSNorm


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for Vision Transformer.
    
    Divides image into patches and projects to embedding space.
    Includes position embeddings for spatial information.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch projection
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Optional MHC enhancement
        if use_mhc:
            self.mhc_enhance = ManifoldHyperConnection(
                input_dim=embed_dim,
                expansion_rate=2
            )
        else:
            self.mhc_enhance = nn.Identity()
        
        # Position embeddings (learnable)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Class token (for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Layer normalization
        self.norm = RMSNorm(embed_dim) if use_mhc else nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize projection
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Patch embeddings with position info [B, N+1, D]
        """
        B, C, H, W = x.shape
        
        # Project patches
        x = self.projection(x)  # [B, D, H/P, W/P]
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Apply MHC enhancement if enabled
        x = self.mhc_enhance(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Normalize
        x = self.norm(x)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with MHC stabilization.
    
    Standard architecture: MHA → MLP → Residual
    with manifold constraints for stability.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_mhc = use_mhc
        
        # Multi-head attention
        self.attention = MultiHeadManifoldAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_mhc=use_mhc
        )
        
        # Attention normalization
        self.norm1 = RMSNorm(embed_dim) if use_mhc else nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # MLP normalization
        self.norm2 = RMSNorm(embed_dim) if use_mhc else nn.LayerNorm(embed_dim)
        
        # MHC stabilization for residual connections
        if use_mhc:
            self.residual_mhc1 = ManifoldHyperConnection(
                input_dim=embed_dim,
                expansion_rate=2
            )
            self.residual_mhc2 = ManifoldHyperConnection(
                input_dim=embed_dim,
                expansion_rate=2
            )
        else:
            self.residual_mhc1 = nn.Identity()
            self.residual_mhc2 = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, N, D]
            
        Returns:
            Output tensor [B, N, D]
        """
        # === Attention Block ===
        residual = x
        
        # Pre-norm
        x = self.norm1(x)
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # MHC-stabilized residual
        attn_output = self.residual_mhc1(attn_output)
        x = residual + self.dropout(attn_output)
        
        # === MLP Block ===
        residual = x
        
        # Pre-norm
        x = self.norm2(x)
        
        # MLP
        mlp_output = self.mlp(x)
        
        # MHC-stabilized residual
        mlp_output = self.residual_mhc2(mlp_output)
        x = residual + self.dropout(mlp_output)
        
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Complete Vision Transformer Encoder.
    
    Processes image patches through transformer blocks.
    Extracts global context for vision tasks.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_mhc: bool = True,
        num_classes: int = 1000
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_mhc = use_mhc
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_mhc=use_mhc
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_mhc=use_mhc
            )
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = RMSNorm(embed_dim) if use_mhc else nn.LayerNorm(embed_dim)
        
        # Classification head (optional)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize transformer weights."""
        # Already initialized in submodules
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through ViT encoder.
        
        Args:
            x: Input image [B, C, H, W]
            return_features: Whether to return all features
            
        Returns:
            Output features or logits
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N+1, D]
        
        # Store intermediate features if needed
        features = [x] if return_features else None
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            if return_features:
                features.append(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract class token
        cls_token = x[:, 0]  # [B, D]
        
        # Classification (optional)
        output = self.head(cls_token)
        
        if return_features:
            return output, features
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification head.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Feature tensor [B, D]
        """
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]  # Class token


class VisionTransformerDecoder(nn.Module):
    """
    Transformer decoder for dense prediction tasks.
    
    Can be used for:
    - Object detection refinement
    - Semantic segmentation
    - Depth estimation
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_mhc: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_mhc=use_mhc
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # MHC enhancement
        if use_mhc:
            self.mhc_fusion = ManifoldHyperConnection(
                input_dim=embed_dim,
                expansion_rate=2
            )
        else:
            self.mhc_fusion = nn.Identity()
    
    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Query features [B, N, D]
            memory: Optional encoder memory [B, M, D]
            
        Returns:
            Decoded features [B, N, D]
        """
        # Process through decoder blocks
        for block in self.blocks:
            if memory is not None:
                # Cross-attention would go here
                # For simplicity, we just process x
                pass
            x = block(x)
        
        # Final projection and MHC enhancement
        x = self.output_proj(x)
        x = self.mhc_fusion(x)
        
        return x


class HybridVisionEncoder(nn.Module):
    """
    Hybrid encoder combining CNN and Transformer.
    
    Uses CNN for local feature extraction and ViT for global context.
    Optimized for robotic vision tasks.
    """
    
    def __init__(
        self,
        cnn_channels: int = 512,
        vit_embed_dim: int = 256,
        vit_depth: int = 6,
        vit_num_heads: int = 8,
        use_mhc: bool = True
    ):
        super().__init__()
        
        # Bridge from CNN to ViT
        self.cnn_to_vit = nn.Conv2d(
            cnn_channels, vit_embed_dim,
            kernel_size=1
        )
        
        # Position embeddings for ViT input
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 256, vit_embed_dim)  # Fixed size for 16x16 grid
        )
        
        # ViT encoder
        self.vit_encoder = VisionTransformerEncoder(
            image_size=16,  # CNN feature map size
            patch_size=1,   # Treat each position as patch
            in_channels=vit_embed_dim,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=4.0,
            dropout=0.1,
            use_mhc=use_mhc,
            num_classes=0  # No classification head
        )
        
        # Feature fusion (ViT → CNN)
        self.vit_to_cnn = nn.Conv2d(
            vit_embed_dim, cnn_channels,
            kernel_size=1
        )
        
        # MHC for feature fusion
        if use_mhc:
            self.fusion_mhc = ManifoldHyperConnection(
                input_dim=cnn_channels,
                expansion_rate=2
            )
        else:
            self.fusion_mhc = nn.Identity()
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Enhance CNN features with ViT global context.
        
        Args:
            cnn_features: CNN features [B, C, H, W]
            
        Returns:
            Enhanced features [B, C, H, W]
        """
        B, C, H, W = cnn_features.shape
        
        # === CNN → ViT ===
        # Project to ViT dimension
        vit_input = self.cnn_to_vit(cnn_features)  # [B, D_vit, H, W]
        
        # Flatten spatial dimensions
        vit_input = vit_input.flatten(2).transpose(1, 2)  # [B, H*W, D_vit]
        
        # Add position embeddings
        if H * W == 256:  # 16x16 grid
            vit_input = vit_input + self.pos_embed
        else:
            # Interpolate position embeddings
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=(H * W,),
                mode='linear'
            ).transpose(1, 2)
            vit_input = vit_input + pos_embed
        
        # Reshape for ViT (add fake height/width dimension)
        vit_input = vit_input.unsqueeze(1)  # [B, 1, H*W, D_vit]
        vit_input = vit_input.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, D_vit, H, W]
        
        # Process through ViT
        vit_features = self.vit_encoder(vit_input)  # [B, D_vit]
        
        # === ViT → CNN ===
        # Expand ViT features spatially
        vit_features = vit_features.unsqueeze(-1).unsqueeze(-1)  # [B, D_vit, 1, 1]
        vit_features = vit_features.expand(-1, -1, H, W)  # [B, D_vit, H, W]
        
        # Project back to CNN dimension
        vit_enhanced = self.vit_to_cnn(vit_features)  # [B, C, H, W]
        
        # Fuse with original CNN features
        fused = cnn_features + vit_enhanced
        fused = self.fusion_mhc(fused)
        
        return fused