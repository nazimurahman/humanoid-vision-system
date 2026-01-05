# src/models/__init__.py

"""
Hybrid Vision System Models Package.

This package contains all model components for the manifold-constrained
hyper-connection vision system for humanoid robots.
"""

from .manifold_layers import (
    SinkhornKnoppProjection,
    ManifoldHyperConnection,
    MultiHeadManifoldAttention,
    RMSNorm
)

from .vision_backbone import (
    ConvMHCLayer,
    ResidualMHCLayer,
    HybridVisionBackbone
)

from .vit_encoder_decoder import (
    PatchEmbedding,
    TransformerEncoderBlock,
    VisionTransformerEncoder,
    VisionTransformerDecoder,
    HybridVisionEncoder
)

from .yolo_head import (
    YOLOAnchorGenerator,
    YOLOPredictionHead,
    YOLODecoder,
    YOLOLoss,
    YOLODetectionHead
)

from .feature_fusion import (
    FeaturePyramidNetwork,
    MultiScaleFeatureFusion,
    CrossScaleAttention,
    AdaptiveFeatureFusion
)

from .rag_module import (
    KnowledgeBase,
    RAGVisionKnowledge,
    KnowledgeAwareDetection
)

from .hybrid_vision import (
    HybridVisionSystem,
    LightweightHybridVision,
    ProductionHybridVision
)

# Version
__version__ = "1.0.0"

# Export all components
__all__ = [
    # Manifold Layers
    "SinkhornKnoppProjection",
    "ManifoldHyperConnection",
    "MultiHeadManifoldAttention",
    "RMSNorm",
    
    # Vision Backbone
    "ConvMHCLayer",
    "ResidualMHCLayer",
    "HybridVisionBackbone",
    
    # ViT Encoder/Decoder
    "PatchEmbedding",
    "TransformerEncoderBlock",
    "VisionTransformerEncoder",
    "VisionTransformerDecoder",
    "HybridVisionEncoder",
    
    # YOLO Detection
    "YOLOAnchorGenerator",
    "YOLOPredictionHead",
    "YOLODecoder",
    "YOLOLoss",
    "YOLODetectionHead",
    
    # Feature Fusion
    "FeaturePyramidNetwork",
    "MultiScaleFeatureFusion",
    "CrossScaleAttention",
    "AdaptiveFeatureFusion",
    
    # RAG Module
    "KnowledgeBase",
    "RAGVisionKnowledge",
    "KnowledgeAwareDetection",
    
    # Complete Systems
    "HybridVisionSystem",
    "LightweightHybridVision",
    "ProductionHybridVision",
]