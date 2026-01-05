"""
Model configuration for the Hybrid Vision System.
Contains all hyperparameters for the vision model architecture.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum
import math
from .base_config import BaseConfig, DeviceType, PrecisionType

class BackboneType(Enum):
    """Vision backbone types."""
    HYBRID_CNN = "hybrid_cnn"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    MOBILENET = "mobilenet"
    CUSTOM = "custom"

class FusionType(Enum):
    """Feature fusion methods."""
    FPN = "fpn"  # Feature Pyramid Network
    PAN = "pan"  # Path Aggregation Network
    BIFPN = "bifpn"  # Bi-directional FPN
    SIMPLE = "simple"
    ATTENTION = "attention"

class HeadType(Enum):
    """Detection head types."""
    YOLO = "yolo"
    FCOS = "fcos"
    RETINANET = "retinanet"
    CENTERNET = "centernet"
    DETR = "detr"

class ActivationType(Enum):
    """Activation functions."""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    SILU = "silu"

@dataclass
class MHCConfig:
    """Manifold-Constrained Hyper-Connection configuration."""
    
    enabled: bool = True
    """Enable MHC layers."""
    
    expansion_rate: int = 4
    """Expansion rate for hyper-connections."""
    
    num_iterations: int = 20
    """Number of Sinkhorn-Knopp iterations."""
    
    epsilon: float = 1e-8
    """Small epsilon for numerical stability."""
    
    alpha: float = 0.01
    """Small initialization factor."""
    
    use_mixed_precision: bool = True
    """Use mixed precision in MHC layers."""
    
    monitor_eigenvalues: bool = True
    """Monitor eigenvalues for stability."""
    
    gradient_clip_mhc: float = 0.5
    """Gradient clipping specific for MHC layers."""
    
    # Stability monitoring
    track_signal_ratio: bool = True
    """Track input/output signal ratio."""
    
    max_eigenvalue_threshold: float = 1.1
    """Maximum allowed eigenvalue (should be ≤ 1)."""
    
    def validate(self):
        """Validate MHC configuration."""
        if self.expansion_rate < 1:
            raise ValueError("expansion_rate must be at least 1")
        
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        
        if self.gradient_clip_mhc <= 0:
            raise ValueError("gradient_clip_mhc must be positive")
        
        if self.max_eigenvalue_threshold < 1.0:
            warnings.warn("max_eigenvalue_threshold should be ≥ 1.0 for stability")

@dataclass
class BackboneConfig:
    """Backbone configuration."""
    
    type: BackboneType = BackboneType.HYBRID_CNN
    """Type of backbone architecture."""
    
    # Hybrid CNN Backbone
    input_channels: int = 3
    """Number of input channels (RGB=3)."""
    
    base_channels: int = 32
    """Base number of channels."""
    
    num_blocks: List[int] = field(default_factory=lambda: [2, 3, 4, 2])
    """Number of blocks in each stage."""
    
    use_mhc_in_backbone: bool = True
    """Use MHC layers in backbone."""
    
    # Advanced backbone settings
    stem_kernel_size: int = 3
    """Kernel size for stem convolution."""
    
    stem_stride: int = 2
    """Stride for stem convolution."""
    
    activation: ActivationType = ActivationType.SILU
    """Activation function."""
    
    use_se: bool = True
    """Use Squeeze-and-Excitation blocks."""
    
    se_ratio: float = 0.25
    """Squeeze-and-Excitation reduction ratio."""
    
    dropout_rate: float = 0.0
    """Dropout rate in backbone."""
    
    # Output channels for each stage
    @property
    def stage_channels(self) -> List[int]:
        """Get output channels for each stage."""
        channels = []
        current = self.base_channels
        
        for i, num_blocks in enumerate(self.num_blocks):
            if i > 0:
                current = current * 2
            channels.append(current)
        
        return channels
    
    @property
    def output_channels(self) -> List[int]:
        """Get output channels for FPN (last 3 stages)."""
        return self.stage_channels[-3:]  # Last 3 stages for multi-scale
    
    def validate(self):
        """Validate backbone configuration."""
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        
        if len(self.num_blocks) < 2:
            raise ValueError("Need at least 2 stages in backbone")
        
        if self.stem_kernel_size % 2 == 0:
            raise ValueError("stem_kernel_size should be odd for symmetric padding")
        
        if self.stem_stride <= 0:
            raise ValueError("stem_stride must be positive")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        if not 0 < self.se_ratio <= 1:
            raise ValueError("se_ratio must be between 0 and 1")

@dataclass
class ViTConfig:
    """Vision Transformer configuration."""
    
    enabled: bool = True
    """Enable Vision Transformer."""
    
    embed_dim: int = 256
    """Embedding dimension."""
    
    depth: int = 6
    """Number of transformer blocks."""
    
    num_heads: int = 8
    """Number of attention heads."""
    
    mlp_ratio: float = 4.0
    """MLP expansion ratio."""
    
    qkv_bias: bool = True
    """Add bias to QKV projection."""
    
    drop_rate: float = 0.0
    """Dropout rate."""
    
    attn_drop_rate: float = 0.0
    """Attention dropout rate."""
    
    drop_path_rate: float = 0.1
    """Stochastic depth rate."""
    
    # Patch embedding
    patch_size: int = 16
    """Patch size for ViT."""
    
    # Position embedding
    use_abs_pos: bool = True
    """Use absolute position embeddings."""
    
    use_rel_pos: bool = False
    """Use relative position embeddings."""
    
    # ViT output
    global_pool: str = 'avg'  # 'avg', 'max', 'token'
    """Global pooling method."""
    
    def validate(self):
        """Validate ViT configuration."""
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        
        if not 0 <= self.drop_rate <= 1:
            raise ValueError("drop_rate must be between 0 and 1")
        
        if not 0 <= self.attn_drop_rate <= 1:
            raise ValueError("attn_drop_rate must be between 0 and 1")
        
        if not 0 <= self.drop_path_rate <= 1:
            raise ValueError("drop_path_rate must be between 0 and 1")
        
        if self.global_pool not in ['avg', 'max', 'token']:
            raise ValueError("global_pool must be 'avg', 'max', or 'token'")

@dataclass
class FusionConfig:
    """Feature fusion configuration."""
    
    type: FusionType = FusionType.FPN
    """Type of feature fusion."""
    
    channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    """Input channels for each scale."""
    
    use_mhc: bool = True
    """Use MHC in fusion."""
    
    # FPN specific
    fpn_channels: int = 256
    """Number of channels in FPN."""
    
    # PAN specific
    pan_channels: int = 256
    """Number of channels in PAN."""
    
    # Attention fusion
    use_attention: bool = False
    """Use attention in fusion."""
    
    attention_heads: int = 8
    """Number of attention heads."""
    
    def validate(self):
        """Validate fusion configuration."""
        if len(self.channels) < 2:
            raise ValueError("Need at least 2 channels for fusion")
        
        if self.fpn_channels <= 0:
            raise ValueError("fpn_channels must be positive")
        
        if self.pan_channels <= 0:
            raise ValueError("pan_channels must be positive")
        
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")

@dataclass
class DetectionHeadConfig:
    """Detection head configuration."""
    
    type: HeadType = HeadType.YOLO
    """Type of detection head."""
    
    num_classes: int = 80
    """Number of object classes."""
    
    # YOLO specific
    anchors: List[List[Tuple[int, int]]] = field(default_factory=lambda: [
        [(10, 13), (16, 30), (33, 23)],    # Small
        [(30, 61), (62, 45), (59, 119)],   # Medium
        [(116, 90), (156, 198), (373, 326)] # Large
    ])
    """Anchor boxes for YOLO."""
    
    num_anchors: int = 3
    """Number of anchors per grid cell."""
    
    # General head settings
    hidden_dim: int = 256
    """Hidden dimension in detection head."""
    
    num_layers: int = 3
    """Number of layers in detection head."""
    
    use_mhc: bool = True
    """Use MHC in detection head."""
    
    # Loss weights
    box_loss_weight: float = 5.0
    """Weight for bounding box loss."""
    
    obj_loss_weight: float = 1.0
    """Weight for objectness loss."""
    
    cls_loss_weight: float = 1.0
    """Weight for classification loss."""
    
    # NMS settings
    nms_iou_threshold: float = 0.5
    """IoU threshold for NMS."""
    
    nms_score_threshold: float = 0.25
    """Score threshold for NMS."""
    
    max_detections: int = 300
    """Maximum number of detections per image."""
    
    def validate(self):
        """Validate detection head configuration."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        
        if self.num_anchors <= 0:
            raise ValueError("num_anchors must be positive")
        
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        if not 0 <= self.nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold must be between 0 and 1")
        
        if not 0 <= self.nms_score_threshold <= 1:
            raise ValueError("nms_score_threshold must be between 0 and 1")
        
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        
        # Validate anchors
        for scale_anchors in self.anchors:
            for anchor in scale_anchors:
                if len(anchor) != 2:
                    raise ValueError("Each anchor must be (width, height)")
                if anchor[0] <= 0 or anchor[1] <= 0:
                    raise ValueError("Anchor dimensions must be positive")

@dataclass
class RAGConfig:
    """Retrieval-Augmented Generation configuration."""
    
    enabled: bool = False
    """Enable RAG module."""
    
    knowledge_dim: int = 512
    """Dimension of knowledge embeddings."""
    
    visual_dim: int = 256
    """Dimension of visual features."""
    
    num_retrievals: int = 5
    """Number of knowledge items to retrieve."""
    
    knowledge_base_size: int = 10000
    """Maximum size of knowledge base."""
    
    retrieval_method: str = "cosine"  # "cosine", "dot", "l2"
    """Method for similarity calculation."""
    
    fusion_method: str = "attention"  # "concat", "add", "attention"
    """Method for fusing visual and knowledge features."""
    
    # Knowledge base paths
    knowledge_base_path: Optional[str] = None
    """Path to knowledge base file."""
    
    embeddings_path: Optional[str] = None
    """Path to precomputed embeddings."""
    
    def validate(self):
        """Validate RAG configuration."""
        if self.knowledge_dim <= 0:
            raise ValueError("knowledge_dim must be positive")
        
        if self.visual_dim <= 0:
            raise ValueError("visual_dim must be positive")
        
        if self.num_retrievals <= 0:
            raise ValueError("num_retrievals must be positive")
        
        if self.knowledge_base_size <= 0:
            raise ValueError("knowledge_base_size must be positive")
        
        if self.retrieval_method not in ["cosine", "dot", "l2"]:
            raise ValueError("retrieval_method must be 'cosine', 'dot', or 'l2'")
        
        if self.fusion_method not in ["concat", "add", "attention"]:
            raise ValueError("fusion_method must be 'concat', 'add', or 'attention'")

@dataclass
class ModelConfig(BaseConfig):
    """
    Complete model configuration for Hybrid Vision System.
    """
    
    # =============== MODEL ARCHITECTURE ===============
    model_name: str = "HybridVisionSystem"
    """Name of the model."""
    
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    """Backbone configuration."""
    
    vit: ViTConfig = field(default_factory=ViTConfig)
    """Vision Transformer configuration."""
    
    fusion: FusionConfig = field(default_factory=FusionConfig)
    """Feature fusion configuration."""
    
    detection_head: DetectionHeadConfig = field(default_factory=DetectionHeadConfig)
    """Detection head configuration."""
    
    mhc: MHCConfig = field(default_factory=MHCConfig)
    """Manifold-Constrained Hyper-Connection configuration."""
    
    rag: RAGConfig = field(default_factory=RAGConfig)
    """RAG configuration."""
    
    # =============== INPUT SETTINGS ===============
    input_height: int = 416
    """Input image height."""
    
    input_width: int = 416
    """Input image width."""
    
    input_channels: int = 3
    """Input image channels (RGB=3)."""
    
    # =============== OUTPUT SETTINGS ===============
    output_stride: int = 32
    """Output stride of the model."""
    
    # =============== ADVANCED MODEL SETTINGS ===============
    use_batch_norm: bool = True
    """Use batch normalization."""
    
    use_layer_norm: bool = False
    """Use layer normalization."""
    
    use_instance_norm: bool = False
    """Use instance normalization."""
    
    norm_momentum: float = 0.1
    """Momentum for normalization layers."""
    
    norm_epsilon: float = 1e-5
    """Epsilon for normalization layers."""
    
    # =============== INITIALIZATION ===============
    init_method: str = "kaiming_normal"  # "kaiming_normal", "xavier_uniform", "trunc_normal"
    """Weight initialization method."""
    
    init_gain: float = 1.0
    """Gain for initialization."""
    
    # =============== PARAMETER COUNT ===============
    @property
    def estimated_parameters(self) -> Dict[str, int]:
        """Estimate number of parameters for each component."""
        params = {}
        
        # Backbone estimate
        backbone_params = 0
        channels = self.backbone.base_channels
        for i, num_blocks in enumerate(self.backbone.num_blocks):
            if i > 0:
                channels *= 2
            # Rough estimate: each block has 2 conv layers
            backbone_params += num_blocks * 2 * (channels ** 2) * 9  # 3x3 kernel
        
        params['backbone'] = backbone_params
        
        # ViT estimate
        if self.vit.enabled:
            vit_params = 0
            # Patch embedding
            vit_params += 3 * self.vit.embed_dim * (self.vit.patch_size ** 2)
            # Transformer blocks
            for _ in range(self.vit.depth):
                # Attention
                vit_params += 3 * (self.vit.embed_dim ** 2)  # QKV
                vit_params += (self.vit.embed_dim ** 2)  # projection
                # MLP
                mlp_dim = int(self.vit.embed_dim * self.vit.mlp_ratio)
                vit_params += self.vit.embed_dim * mlp_dim * 2
            
            params['vit'] = vit_params
        
        # Fusion estimate
        fusion_params = 0
        for channels in self.fusion.channels:
            fusion_params += channels * self.fusion.fpn_channels * 3  # Rough estimate
        params['fusion'] = fusion_params
        
        # Detection head estimate
        head_params = 0
        head_params += self.fusion.fpn_channels * self.detection_head.hidden_dim
        head_params += self.detection_head.num_layers * (self.detection_head.hidden_dim ** 2)
        head_params += self.detection_head.hidden_dim * (
            5 + self.detection_head.num_classes
        ) * self.detection_head.num_anchors
        params['detection_head'] = head_params
        
        # Total estimate
        params['total'] = sum(params.values())
        
        return params
    
    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()
        self._validate_model_config()
    
    def _validate_model_config(self):
        """Validate model-specific configuration."""
        # Validate sub-configs
        self.backbone.validate()
        self.vit.validate()
        self.fusion.validate()
        self.detection_head.validate()
        self.mhc.validate()
        self.rag.validate()
        
        # Validate input dimensions
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("Input dimensions must be positive")
        
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        
        if self.output_stride <= 0:
            raise ValueError("output_stride must be positive")
        
        # Validate normalization settings
        if not 0 < self.norm_momentum < 1:
            raise ValueError("norm_momentum must be between 0 and 1")
        
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")
        
        # Check if input is divisible by patch size for ViT
        if self.vit.enabled:
            if self.input_height % self.vit.patch_size != 0:
                raise ValueError(f"input_height {self.input_height} must be divisible by patch_size {self.vit.patch_size}")
            if self.input_width % self.vit.patch_size != 0:
                raise ValueError(f"input_width {self.input_width} must be divisible by patch_size {self.vit.patch_size}")
        
        # Set fusion channels from backbone if not specified
        if not self.fusion.channels:
            self.fusion.channels = self.backbone.output_channels
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get input shape as (channels, height, width)."""
        return (self.input_channels, self.input_height, self.input_width)
    
    def get_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get expected output shapes for different components."""
        shapes = {}
        
        # Backbone output shapes
        for i, channels in enumerate(self.backbone.output_channels):
            scale_factor = 2 ** (len(self.backbone.output_channels) - i)
            height = self.input_height // scale_factor
            width = self.input_width // scale_factor
            shapes[f'backbone_scale_{i}'] = (channels, height, width)
        
        # Detection head output shape
        # Assuming output at each scale
        for i in range(len(self.backbone.output_channels)):
            scale_factor = 2 ** (len(self.backbone.output_channels) - i)
            height = self.input_height // scale_factor
            width = self.input_width // scale_factor
            num_outputs = self.detection_head.num_anchors * (
                5 + self.detection_head.num_classes
            )
            shapes[f'detection_scale_{i}'] = (num_outputs, height, width)
        
        return shapes
    
    def display_detailed(self):
        """Display detailed model configuration."""
        super().display()
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE DETAILS")
        print("="*60)
        
        # Parameter estimates
        params = self.estimated_parameters
        print(f"\nParameter Estimates:")
        print("-" * 40)
        for component, count in params.items():
            if component != 'total':
                print(f"  {component:15s}: {count:,}")
        print(f"  {'total':15s}: {params['total']:,}")
        
        # Output shapes
        shapes = self.get_output_shapes()
        print(f"\nOutput Shapes:")
        print("-" * 40)
        for name, shape in shapes.items():
            print(f"  {name:20s}: {shape}")
        
        # MHC details
        if self.mhc.enabled:
            print(f"\nMHC Configuration:")
            print("-" * 40)
            print(f"  Expansion Rate: {self.mhc.expansion_rate}")
            print(f"  Sinkhorn Iterations: {self.mhc.num_iterations}")
            print(f"  Gradient Clip: {self.mhc.gradient_clip_mhc}")
        
        print("="*60)