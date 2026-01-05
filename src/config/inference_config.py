"""
Inference configuration for the Humanoid Vision System.
Contains all settings for model deployment and inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import torch
from .base_config import BaseConfig, DeviceType, PrecisionType
from .model_config import ModelConfig

class InferenceEngine(Enum):
    """Inference engine types."""
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TRITON = "triton"

class InputFormat(Enum):
    """Input data formats."""
    TENSOR = "tensor"
    NUMPY = "numpy"
    PIL = "pil"
    CV2 = "cv2"
    BYTES = "bytes"
    BASE64 = "base64"

class OutputFormat(Enum):
    """Output data formats."""
    TENSOR = "tensor"
    NUMPY = "numpy"
    DICT = "dict"
    JSON = "json"
    VISUALIZATION = "visualization"

class VisualizationType(Enum):
    """Visualization types."""
    BBOX = "bbox"
    MASK = "mask"
    KEYPOINTS = "keypoints"
    HEATMAP = "heatmap"
    ATTENTION = "attention"

@dataclass
class PreprocessingConfig:
    """Preprocessing configuration for inference."""
    
    # Resize settings
    resize_method: str = "letterbox"  # "letterbox", "resize", "center_crop"
    """Method for resizing images."""
    
    keep_aspect_ratio: bool = True
    """Keep aspect ratio when resizing."""
    
    pad_color: Tuple[int, int, int] = (114, 114, 114)
    """Padding color for letterbox resizing."""
    
    # Normalization
    normalization: bool = True
    """Apply normalization."""
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    """Mean for normalization."""
    
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    """Std for normalization."""
    
    # Color conversion
    bgr_to_rgb: bool = True
    """Convert BGR to RGB (for OpenCV inputs)."""
    
    # Preprocessing pipeline
    pipeline: List[str] = field(default_factory=lambda: [
        "resize", "normalize", "to_tensor"
    ])
    """Preprocessing pipeline steps."""
    
    # Augmentation for TTA
    tta_transforms: List[str] = field(default_factory=list)
    """Transforms for test-time augmentation."""
    
    def validate(self):
        """Validate preprocessing configuration."""
        if len(self.mean) != 3:
            raise ValueError("mean must have 3 values")
        
        if len(self.std) != 3:
            raise ValueError("std must have 3 values")
        
        if self.resize_method not in ["letterbox", "resize", "center_crop"]:
            raise ValueError("resize_method must be one of: letterbox, resize, center_crop")
        
        if len(self.pad_color) != 3:
            raise ValueError("pad_color must have 3 values (RGB)")
        
        for value in self.pad_color:
            if not 0 <= value <= 255:
                raise ValueError("pad_color values must be between 0 and 255")

@dataclass
class PostprocessingConfig:
    """Postprocessing configuration for inference."""
    
    # Detection filtering
    confidence_threshold: float = 0.25
    """Minimum confidence score for detections."""
    
    nms_threshold: float = 0.45
    """IoU threshold for Non-Maximum Suppression."""
    
    nms_method: str = "standard"  # "standard", "soft", "fast", "cluster"
    """NMS method."""
    
    max_detections: int = 300
    """Maximum number of detections per image."""
    
    # Output formatting
    output_format: OutputFormat = OutputFormat.DICT
    """Format of output predictions."""
    
    include_scores: bool = True
    """Include confidence scores in output."""
    
    include_classes: bool = True
    """Include class labels in output."""
    
    include_masks: bool = False
    """Include segmentation masks in output."""
    
    include_keypoints: bool = False
    """Include keypoints in output."""
    
    # Coordinate conversion
    convert_to_original: bool = True
    """Convert coordinates back to original image size."""
    
    relative_coordinates: bool = False
    """Use relative coordinates (0-1) instead of absolute."""
    
    # Class mapping
    class_names: Optional[List[str]] = None
    """List of class names for mapping."""
    
    class_filter: Optional[List[int]] = None
    """Filter specific classes in output."""
    
    # Score calibration
    temperature: float = 1.0
    """Temperature for score calibration."""
    
    def validate(self):
        """Validate postprocessing configuration."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        
        if self.nms_method not in ["standard", "soft", "fast", "cluster"]:
            raise ValueError("nms_method must be one of: standard, soft, fast, cluster")
        
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    
    enabled: bool = True
    """Enable visualization."""
    
    types: List[VisualizationType] = field(default_factory=lambda: [VisualizationType.BBOX])
    """Types of visualizations to generate."""
    
    # Bounding box settings
    bbox_thickness: int = 2
    """Thickness of bounding box lines."""
    
    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    """Color of bounding boxes (RGB)."""
    
    show_labels: bool = True
    """Show class labels on bounding boxes."""
    
    show_scores: bool = True
    """Show confidence scores on bounding boxes."""
    
    label_font_size: float = 0.5
    """Font size for labels."""
    
    label_thickness: int = 1
    """Thickness of label text."""
    
    label_color: Tuple[int, int, int] = (0, 0, 0)
    """Color of label text (RGB)."""
    
    label_background: bool = True
    """Add background to labels for better visibility."""
    
    # Mask settings
    mask_alpha: float = 0.5
    """Alpha transparency for masks."""
    
    mask_colors: Optional[List[Tuple[int, int, int]]] = None
    """Colors for different classes in masks."""
    
    # Heatmap settings
    heatmap_colormap: str = "jet"  # "viridis", "plasma", "inferno", "magma", "cividis"
    """Colormap for heatmaps."""
    
    heatmap_alpha: float = 0.5
    """Alpha transparency for heatmaps."""
    
    # Output settings
    save_visualizations: bool = True
    """Save visualizations to disk."""
    
    output_dir: str = "visualizations"
    """Directory for saving visualizations."""
    
    output_format: str = "png"  # "png", "jpg", "pdf", "svg"
    """Format for saved visualizations."""
    
    display: bool = False
    """Display visualizations interactively."""
    
    def validate(self):
        """Validate visualization configuration."""
        if not 0 <= self.mask_alpha <= 1:
            raise ValueError("mask_alpha must be between 0 and 1")
        
        if not 0 <= self.heatmap_alpha <= 1:
            raise ValueError("heatmap_alpha must be between 0 and 1")
        
        if self.bbox_thickness <= 0:
            raise ValueError("bbox_thickness must be positive")
        
        if self.label_thickness <= 0:
            raise ValueError("label_thickness must be positive")
        
        if self.label_font_size <= 0:
            raise ValueError("label_font_size must be positive")
        
        for color in [self.bbox_color, self.label_color]:
            if len(color) != 3:
                raise ValueError("Colors must have 3 values (RGB)")
            for value in color:
                if not 0 <= value <= 255:
                    raise ValueError("Color values must be between 0 and 255")

@dataclass
class APIConfig:
    """API server configuration."""
    
    enabled: bool = False
    """Enable API server."""
    
    host: str = "0.0.0.0"
    """Host address for API server."""
    
    port: int = 8000
    """Port for API server."""
    
    workers: int = 1
    """Number of worker processes."""
    
    # API endpoints
    endpoints: List[str] = field(default_factory=lambda: [
        "/predict",
        "/batch_predict",
        "/health",
        "/metrics",
        "/info"
    ])
    """Available API endpoints."""
    
    # Rate limiting
    rate_limit: bool = True
    """Enable rate limiting."""
    
    requests_per_second: int = 10
    """Maximum requests per second."""
    
    # Authentication
    auth_enabled: bool = False
    """Enable authentication."""
    
    auth_token: Optional[str] = None
    """Authentication token."""
    
    # CORS
    cors_enabled: bool = True
    """Enable CORS."""
    
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    """Allowed CORS origins."""
    
    # Timeouts
    timeout: int = 30
    """Request timeout in seconds."""
    
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    """Maximum request size in bytes."""
    
    def validate(self):
        """Validate API configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        if self.max_request_size <= 0:
            raise ValueError("max_request_size must be positive")

@dataclass
class GRPCConfig:
    """gRPC server configuration for robotic deployment."""
    
    enabled: bool = True
    """Enable gRPC server."""
    
    host: str = "0.0.0.0"
    """Host address for gRPC server."""
    
    port: int = 50051
    """Port for gRPC server."""
    
    max_message_length: int = 100 * 1024 * 1024  # 100MB
    """Maximum message length."""
    
    max_workers: int = 10
    """Maximum number of gRPC workers."""
    
    # Streaming
    streaming_enabled: bool = True
    """Enable streaming for camera feeds."""
    
    stream_buffer_size: int = 10
    """Buffer size for streaming."""
    
    # Robot-specific settings
    robot_id: Optional[str] = None
    """Robot identifier."""
    
    robot_type: str = "humanoid"
    """Type of robot."""
    
    def validate(self):
        """Validate gRPC configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if self.max_message_length <= 0:
            raise ValueError("max_message_length must be positive")
        
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.stream_buffer_size <= 0:
            raise ValueError("stream_buffer_size must be positive")

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Batching
    batch_size: int = 1
    """Batch size for inference."""
    
    dynamic_batching: bool = True
    """Enable dynamic batching."""
    
    max_batch_size: int = 16
    """Maximum batch size."""
    
    batch_timeout: float = 0.1
    """Timeout for batching in seconds."""
    
    # Memory optimization
    memory_efficient: bool = True
    """Enable memory-efficient inference."""
    
    tensor_float_32: bool = True
    """Enable TF32 precision for faster inference."""
    
    workspace_size: int = 1024 * 1024 * 1024  # 1GB
    """Workspace size for TensorRT/other engines."""
    
    # Parallel processing
    num_streams: int = 1
    """Number of CUDA streams for parallel processing."""
    
    async_inference: bool = True
    """Enable asynchronous inference."""
    
    # Caching
    cache_enabled: bool = True
    """Enable caching of inference results."""
    
    cache_size: int = 1000
    """Maximum cache size."""
    
    cache_ttl: int = 3600  # 1 hour
    """Cache time-to-live in seconds."""
    
    # Profiling
    profiling_enabled: bool = False
    """Enable performance profiling."""
    
    profile_steps: int = 100
    """Number of steps to profile."""
    
    def validate(self):
        """Validate performance configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        
        if self.batch_timeout <= 0:
            raise ValueError("batch_timeout must be positive")
        
        if self.workspace_size <= 0:
            raise ValueError("workspace_size must be positive")
        
        if self.num_streams <= 0:
            raise ValueError("num_streams must be positive")
        
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        
        if self.profile_steps <= 0:
            raise ValueError("profile_steps must be positive")

@dataclass
class InferenceConfig(BaseConfig):
    """
    Complete inference configuration for Hybrid Vision System.
    """
    
    # =============== MODEL DEPLOYMENT ===============
    model_path: str = "models/hybrid_vision.pt"
    """Path to trained model file."""
    
    engine: InferenceEngine = InferenceEngine.PYTORCH
    """Inference engine to use."""
    
    model_format: str = "pytorch"  # "pytorch", "torchscript", "onnx", "tensorrt"
    """Format of the model file."""
    
    # =============== INPUT/OUTPUT SETTINGS ===============
    input_format: InputFormat = InputFormat.CV2
    """Format of input data."""
    
    output_format: OutputFormat = OutputFormat.DICT
    """Format of output data."""
    
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    """Preprocessing configuration."""
    
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    """Postprocessing configuration."""
    
    # =============== VISUALIZATION ===============
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    """Visualization configuration."""
    
    # =============== SERVING ===============
    api: APIConfig = field(default_factory=APIConfig)
    """API server configuration."""
    
    grpc: GRPCConfig = field(default_factory=GRPCConfig)
    """gRPC server configuration."""
    
    # =============== PERFORMANCE ===============
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    """Performance configuration."""
    
    # =============== ROBOTIC DEPLOYMENT ===============
    robot_interface: bool = True
    """Enable robot-specific interface."""
    
    camera_source: str = "0"  # "0" for default camera, path for video file, URL for stream
    """Camera source for robotic deployment."""
    
    frame_rate: int = 30
    """Target frame rate for robotic deployment."""
    
    real_time: bool = True
    """Enable real-time processing."""
    
    latency_target: float = 50.0  # milliseconds
    """Target latency for inference."""
    
    # =============== MONITORING ===============
    monitoring_enabled: bool = True
    """Enable inference monitoring."""
    
    metrics_port: int = 9090
    """Port for metrics endpoint."""
    
    health_check_interval: int = 30
    """Health check interval in seconds."""
    
    # =============== LOGGING ===============
    log_predictions: bool = False
    """Log prediction results."""
    
    log_level: str = "INFO"
    """Logging level for inference."""
    
    # =============== SAFETY ===============
    safety_checks: bool = True
    """Enable safety checks."""
    
    max_image_size: Tuple[int, int] = (3840, 2160)  # 4K
    """Maximum image size allowed."""
    
    max_batch_memory: int = 4 * 1024 * 1024 * 1024  # 4GB
    """Maximum memory for batch processing."""
    
    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()
        self._validate_inference_config()
    
    def _validate_inference_config(self):
        """Validate inference-specific configuration."""
        # Validate sub-configs
        self.preprocessing.validate()
        self.postprocessing.validate()
        self.visualization.validate()
        self.api.validate()
        self.grpc.validate()
        self.performance.validate()
        
        # Validate main settings
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        
        if self.latency_target <= 0:
            raise ValueError("latency_target must be positive")
        
        if self.metrics_port <= 0 or self.metrics_port > 65535:
            raise ValueError("metrics_port must be between 1 and 65535")
        
        if self.health_check_interval <= 0:
            raise ValueError("health_check_interval must be positive")
        
        if self.max_image_size[0] <= 0 or self.max_image_size[1] <= 0:
            raise ValueError("max_image_size dimensions must be positive")
        
        if self.max_batch_memory <= 0:
            raise ValueError("max_batch_memory must be positive")
    
    def get_engine_settings(self) -> Dict[str, Any]:
        """Get settings specific to the inference engine."""
        settings = {
            'engine': self.engine.value,
            'model_format': self.model_format,
            'precision': self.precision.value,
            'device': self.device.value,
        }
        
        if self.engine == InferenceEngine.TENSORRT:
            settings.update({
                'workspace_size': self.performance.workspace_size,
                'fp16_mode': self.precision == PrecisionType.FP16,
                'int8_mode': self.precision == PrecisionType.INT8,
            })
        elif self.engine == InferenceEngine.ONNXRUNTIME:
            settings.update({
                'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                'session_options': {
                    'intra_op_num_threads': 1,
                    'inter_op_num_threads': 1,
                }
            })
        elif self.engine == InferenceEngine.TRITON:
            settings.update({
                'model_repository': 'models/triton',
                'max_batch_size': self.performance.max_batch_size,
            })
        
        return settings
    
    def get_inference_pipeline(self) -> List[Dict[str, Any]]:
        """Get the complete inference pipeline configuration."""
        pipeline = []
        
        # Input handling
        pipeline.append({
            'step': 'input',
            'type': self.input_format.value,
            'config': {}
        })
        
        # Preprocessing
        pipeline.append({
            'step': 'preprocess',
            'config': {
                'resize_method': self.preprocessing.resize_method,
                'keep_aspect_ratio': self.preprocessing.keep_aspect_ratio,
                'normalization': self.preprocessing.normalization,
                'mean': self.preprocessing.mean,
                'std': self.preprocessing.std,
            }
        })
        
        # Inference
        pipeline.append({
            'step': 'inference',
            'engine': self.engine.value,
            'config': self.get_engine_settings()
        })
        
        # Postprocessing
        pipeline.append({
            'step': 'postprocess',
            'config': {
                'confidence_threshold': self.postprocessing.confidence_threshold,
                'nms_threshold': self.postprocessing.nms_threshold,
                'max_detections': self.postprocessing.max_detections,
                'output_format': self.postprocessing.output_format.value,
            }
        })
        
        # Visualization (optional)
        if self.visualization.enabled:
            pipeline.append({
                'step': 'visualize',
                'config': {
                    'types': [t.value for t in self.visualization.types],
                    'save': self.visualization.save_visualizations,
                    'display': self.visualization.display,
                }
            })
        
        return pipeline
    
    def display_inference_summary(self):
        """Display detailed inference configuration summary."""
        super().display()
        
        print("\n" + "="*60)
        print("INFERENCE CONFIGURATION DETAILS")
        print("="*60)
        
        # Model info
        print(f"\nModel Deployment:")
        print("-" * 40)
        print(f"  Model path: {self.model_path}")
        print(f"  Engine: {self.engine.value}")
        print(f"  Format: {self.model_format}")
        print(f"  Device: {self.device.value}")
        print(f"  Precision: {self.precision.value}")
        
        # Performance info
        print(f"\nPerformance:")
        print("-" * 40)
        print(f"  Batch size: {self.performance.batch_size}")
        print(f"  Dynamic batching: {self.performance.dynamic_batching}")
        print(f"  Async inference: {self.performance.async_inference}")
        print(f"  Latency target: {self.latency_target}ms")
        
        # Processing info
        print(f"\nProcessing:")
        print("-" * 40)
        print(f"  Input format: {self.input_format.value}")
        print(f"  Output format: {self.output_format.value}")
        print(f"  Confidence threshold: {self.postprocessing.confidence_threshold}")
        print(f"  NMS threshold: {self.postprocessing.nms_threshold}")
        
        # Serving info
        print(f"\nServing:")
        print("-" * 40)
        print(f"  API enabled: {self.api.enabled}")
        if self.api.enabled:
            print(f"  API host: {self.api.host}:{self.api.port}")
        print(f"  gRPC enabled: {self.grpc.enabled}")
        if self.grpc.enabled:
            print(f"  gRPC host: {self.grpc.host}:{self.grpc.port}")
        
        # Robotic deployment
        print(f"\nRobotic Deployment:")
        print("-" * 40)
        print(f"  Robot interface: {self.robot_interface}")
        print(f"  Camera source: {self.camera_source}")
        print(f"  Frame rate: {self.frame_rate}")
        print(f"  Real-time: {self.real_time}")
        
        # Monitoring
        print(f"\nMonitoring:")
        print("-" * 40)
        print(f"  Monitoring enabled: {self.monitoring_enabled}")
        print(f"  Metrics port: {self.metrics_port}")
        print(f"  Health check interval: {self.health_check_interval}s")
        
        print("="*60)