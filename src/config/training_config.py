"""
Training configuration for the Humanoid Vision System.
Contains all hyperparameters for training and optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import torch
from .base_config import BaseConfig, DeviceType, PrecisionType
from .model_config import ModelConfig

class OptimizerType(Enum):
    """Optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSprop = "rmsprop"
    ADAGRAD = "adagrad"
    LION = "lion"

class SchedulerType(Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    STEP = "step"
    MULTISTEP = "multistep"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    ONE_CYCLE = "one_cycle"
    CONSTANT = "constant"
    WARMUP_COSINE = "warmup_cosine"

class LossType(Enum):
    """Loss function types."""
    YOLO = "yolo"
    FOCAL = "focal"
    SMOOTH_L1 = "smooth_l1"
    CIOU = "ciou"
    DIOU = "diou"
    GIOU = "giou"

class DatasetType(Enum):
    """Dataset types."""
    COCO = "coco"
    VOC = "voc"
    CUSTOM = "custom"
    SYNTHETIC = "synthetic"

class AugmentationType(Enum):
    """Augmentation types."""
    BASIC = "basic"
    ADVANCED = "advanced"
    MOSAIC = "mosaic"
    MIXUP = "mixup"
    CUTMIX = "cutmix"
    AUTOAUGMENT = "autoaugment"
    RANDAUGMENT = "randaugment"

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    type: DatasetType = DatasetType.COCO
    """Type of dataset."""
    
    train_path: str = "datasets/coco/train2017"
    """Path to training data."""
    
    val_path: str = "datasets/coco/val2017"
    """Path to validation data."""
    
    test_path: Optional[str] = None
    """Path to test data."""
    
    annotations_path: str = "datasets/coco/annotations"
    """Path to annotations."""
    
    # Data splitting
    train_split: float = 0.8
    """Fraction of data for training."""
    
    val_split: float = 0.1
    """Fraction of data for validation."""
    
    test_split: float = 0.1
    """Fraction of data for testing."""
    
    # Class information
    class_names: List[str] = field(default_factory=list)
    """List of class names."""
    
    num_classes: int = 80
    """Number of classes."""
    
    # Dataset statistics (for normalization)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    """Mean for normalization (RGB)."""
    
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    """Std for normalization (RGB)."""
    
    # Cache settings
    cache_images: bool = False
    """Cache images in memory."""
    
    cache_dir: Optional[str] = None
    """Directory for caching."""
    
    # Streaming settings
    streaming: bool = False
    """Use streaming dataset."""
    
    stream_buffer_size: int = 1000
    """Buffer size for streaming."""
    
    def validate(self):
        """Validate dataset configuration."""
        if not 0 <= self.train_split <= 1:
            raise ValueError("train_split must be between 0 and 1")
        
        if not 0 <= self.val_split <= 1:
            raise ValueError("val_split must be between 0 and 1")
        
        if not 0 <= self.test_split <= 1:
            raise ValueError("test_split must be between 0 and 1")
        
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Splits must sum to 1.0")
        
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        
        if len(self.mean) != 3:
            raise ValueError("mean must have 3 values (RGB)")
        
        if len(self.std) != 3:
            raise ValueError("std must have 3 values (RGB)")

@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    
    type: AugmentationType = AugmentationType.ADVANCED
    """Type of augmentation."""
    
    # Basic augmentations
    random_flip: bool = True
    """Random horizontal flip."""
    
    flip_probability: float = 0.5
    """Probability of flipping."""
    
    random_rotate: bool = True
    """Random rotation."""
    
    rotate_degrees: float = 10.0
    """Maximum rotation degrees."""
    
    random_scale: bool = True
    """Random scaling."""
    
    scale_range: Tuple[float, float] = (0.8, 1.2)
    """Scale range."""
    
    random_crop: bool = True
    """Random cropping."""
    
    crop_min_scale: float = 0.3
    """Minimum crop scale."""
    
    # Color augmentations
    color_jitter: bool = True
    """Color jitter."""
    
    brightness: float = 0.2
    """Brightness adjustment range."""
    
    contrast: float = 0.2
    """Contrast adjustment range."""
    
    saturation: float = 0.2
    """Saturation adjustment range."""
    
    hue: float = 0.1
    """Hue adjustment range."""
    
    # Advanced augmentations
    mosaic: bool = True
    """Mosaic augmentation."""
    
    mosaic_probability: float = 0.5
    """Probability of applying mosaic."""
    
    mixup: bool = True
    """Mixup augmentation."""
    
    mixup_alpha: float = 0.8
    """Mixup alpha parameter."""
    
    cutmix: bool = False
    """CutMix augmentation."""
    
    cutmix_alpha: float = 1.0
    """CutMix alpha parameter."""
    
    # AutoAugment
    autoaugment: bool = False
    """AutoAugment policy."""
    
    autoaugment_policy: str = "v0"  # "v0", "v1", "v2", "v3"
    """AutoAugment policy version."""
    
    # RandAugment
    randaugment: bool = False
    """RandAugment."""
    
    randaugment_magnitude: int = 9
    """RandAugment magnitude."""
    
    randaugment_num_layers: int = 2
    """Number of RandAugment layers."""
    
    # Test-time augmentations
    tta_enabled: bool = False
    """Test-time augmentation."""
    
    tta_flips: bool = True
    """Flip augmentations for TTA."""
    
    tta_scales: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    """Scales for TTA."""
    
    def validate(self):
        """Validate augmentation configuration."""
        if not 0 <= self.flip_probability <= 1:
            raise ValueError("flip_probability must be between 0 and 1")
        
        if self.rotate_degrees < 0:
            raise ValueError("rotate_degrees must be non-negative")
        
        if self.scale_range[0] <= 0 or self.scale_range[1] <= 0:
            raise ValueError("scale_range values must be positive")
        
        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range[0] must be <= scale_range[1]")
        
        if not 0 <= self.crop_min_scale <= 1:
            raise ValueError("crop_min_scale must be between 0 and 1")
        
        if self.brightness < 0 or self.contrast < 0 or self.saturation < 0:
            raise ValueError("Color jitter parameters must be non-negative")
        
        if not 0 <= self.hue <= 0.5:
            raise ValueError("hue must be between 0 and 0.5")
        
        if not 0 <= self.mosaic_probability <= 1:
            raise ValueError("mosaic_probability must be between 0 and 1")
        
        if self.mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be positive")
        
        if self.cutmix_alpha <= 0:
            raise ValueError("cutmix_alpha must be positive")

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    
    type: OptimizerType = OptimizerType.ADAMW
    """Type of optimizer."""
    
    learning_rate: float = 1e-3
    """Initial learning rate."""
    
    # Adam/AdamW specific
    betas: Tuple[float, float] = (0.9, 0.999)
    """Beta parameters for Adam."""
    
    eps: float = 1e-8
    """Epsilon for numerical stability."""
    
    weight_decay: float = 1e-4
    """Weight decay (L2 regularization)."""
    
    amsgrad: bool = False
    """Use AMSGrad variant of Adam."""
    
    # SGD specific
    momentum: float = 0.9
    """Momentum for SGD."""
    
    nesterov: bool = True
    """Use Nesterov momentum."""
    
    # RMSprop specific
    alpha: float = 0.99
    """Smoothing constant for RMSprop."""
    
    # Gradient clipping
    gradient_clip: bool = True
    """Enable gradient clipping."""
    
    max_grad_norm: float = 1.0
    """Maximum gradient norm."""
    
    clip_type: str = "norm"  # "norm", "value", "adaptive"
    """Type of gradient clipping."""
    
    # Learning rate warmup
    warmup_epochs: int = 5
    """Number of warmup epochs."""
    
    warmup_factor: float = 0.1
    """Initial learning rate multiplier during warmup."""
    
    def validate(self):
        """Validate optimizer configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if not 0 <= self.betas[0] < 1:
            raise ValueError("beta1 must be between 0 and 1")
        
        if not 0 <= self.betas[1] < 1:
            raise ValueError("beta2 must be between 0 and 1")
        
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be between 0 and 1")
        
        if not 0 <= self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        
        if not 0 <= self.warmup_factor <= 1:
            raise ValueError("warmup_factor must be between 0 and 1")

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    
    type: SchedulerType = SchedulerType.COSINE
    """Type of scheduler."""
    
    # Cosine annealing
    t_max: int = 100
    """Maximum number of iterations for cosine annealing."""
    
    eta_min: float = 1e-6
    """Minimum learning rate."""
    
    # Step decay
    step_size: int = 30
    """Period of learning rate decay."""
    
    gamma: float = 0.1
    """Multiplicative factor of learning rate decay."""
    
    # Multi-step decay
    milestones: List[int] = field(default_factory=lambda: [60, 90])
    """Milestones for learning rate decay."""
    
    # Exponential decay
    decay_rate: float = 0.96
    """Decay rate for exponential decay."""
    
    # Plateau
    patience: int = 10
    """Number of epochs with no improvement before reducing LR."""
    
    threshold: float = 1e-4
    """Threshold for measuring the new optimum."""
    
    cooldown: int = 0
    """Number of epochs to wait before resuming normal operation."""
    
    min_lr: float = 1e-6
    """Minimum learning rate for plateau scheduler."""
    
    # One-cycle
    max_lr: float = 1e-2
    """Maximum learning rate for one-cycle scheduler."""
    
    pct_start: float = 0.3
    """Percentage of cycle spent increasing learning rate."""
    
    div_factor: float = 25.0
    """Determines initial learning rate: max_lr/div_factor."""
    
    final_div_factor: float = 1e4
    """Determines minimum learning rate: max_lr/final_div_factor."""
    
    # Cycle length for cosine with restarts
    cycle_mult: float = 1.0
    """Factor by which to increase cycle length after each restart."""
    
    def validate(self):
        """Validate scheduler configuration."""
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")
        
        if self.eta_min <= 0:
            raise ValueError("eta_min must be positive")
        
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be between 0 and 1")
        
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        
        if self.cooldown < 0:
            raise ValueError("cooldown must be non-negative")
        
        if self.min_lr <= 0:
            raise ValueError("min_lr must be positive")
        
        if self.max_lr <= 0:
            raise ValueError("max_lr must be positive")
        
        if not 0 <= self.pct_start <= 1:
            raise ValueError("pct_start must be between 0 and 1")
        
        if self.div_factor <= 0:
            raise ValueError("div_factor must be positive")
        
        if self.final_div_factor <= 0:
            raise ValueError("final_div_factor must be positive")

@dataclass
class LossConfig:
    """Loss function configuration."""
    
    type: LossType = LossType.YOLO
    """Type of loss function."""
    
    # YOLO loss specific
    box_loss_weight: float = 5.0
    """Weight for bounding box loss."""
    
    obj_loss_weight: float = 1.0
    """Weight for objectness loss."""
    
    noobj_loss_weight: float = 0.5
    """Weight for no-object loss."""
    
    cls_loss_weight: float = 1.0
    """Weight for classification loss."""
    
    # Focal loss specific
    focal_alpha: float = 0.25
    """Alpha parameter for focal loss."""
    
    focal_gamma: float = 2.0
    """Gamma parameter for focal loss."""
    
    # IoU loss variants
    iou_type: str = "ciou"  # "iou", "giou", "diou", "ciou"
    """Type of IoU loss."""
    
    # Smooth L1
    beta: float = 1.0
    """Beta parameter for smooth L1 loss."""
    
    # Label smoothing
    label_smoothing: float = 0.0
    """Label smoothing factor."""
    
    # Loss reduction
    reduction: str = "mean"  # "mean", "sum", "none"
    """Loss reduction method."""
    
    # Class balancing
    class_weights: Optional[List[float]] = None
    """Manual class weights for imbalance."""
    
    use_focal: bool = True
    """Use focal loss for classification."""
    
    def validate(self):
        """Validate loss configuration."""
        if self.box_loss_weight <= 0:
            raise ValueError("box_loss_weight must be positive")
        
        if self.obj_loss_weight <= 0:
            raise ValueError("obj_loss_weight must be positive")
        
        if self.noobj_loss_weight <= 0:
            raise ValueError("noobj_loss_weight must be positive")
        
        if self.cls_loss_weight <= 0:
            raise ValueError("cls_loss_weight must be positive")
        
        if not 0 <= self.focal_alpha <= 1:
            raise ValueError("focal_alpha must be between 0 and 1")
        
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        
        if self.iou_type not in ["iou", "giou", "diou", "ciou"]:
            raise ValueError("iou_type must be one of: iou, giou, diou, ciou")
        
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        
        if not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be between 0 and 1")
        
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

@dataclass
class TrainingConfig(BaseConfig):
    """
    Complete training configuration for Hybrid Vision System.
    """
    
    # =============== TRAINING SETTINGS ===============
    num_epochs: int = 100
    """Number of training epochs."""
    
    start_epoch: int = 0
    """Starting epoch (for resuming)."""
    
    checkpoint_interval: int = 5
    """Save checkpoint every N epochs."""
    
    eval_interval: int = 1
    """Evaluate model every N epochs."""
    
    log_interval: int = 10
    """Log training stats every N batches."""
    
    save_best: bool = True
    """Save best model based on validation metric."""
    
    best_metric: str = "val_loss"  # "val_loss", "val_map", "val_precision"
    """Metric for determining best model."""
    
    # =============== DATASET & AUGMENTATION ===============
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset configuration."""
    
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    """Augmentation configuration."""
    
    # =============== OPTIMIZATION ===============
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration."""
    
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler configuration."""
    
    loss: LossConfig = field(default_factory=LossConfig)
    """Loss configuration."""
    
    # =============== DISTRIBUTED TRAINING ===============
    distributed: bool = False
    """Enable distributed training."""
    
    world_size: int = 1
    """Number of processes for distributed training."""
    
    rank: int = 0
    """Process rank for distributed training."""
    
    dist_backend: str = "nccl"  # "nccl", "gloo", "mpi"
    """Backend for distributed training."""
    
    dist_url: str = "tcp://127.0.0.1:23456"
    """URL used to set up distributed training."""
    
    # =============== EARLY STOPPING ===============
    early_stopping: bool = True
    """Enable early stopping."""
    
    early_stopping_patience: int = 20
    """Patience for early stopping."""
    
    early_stopping_min_delta: float = 1e-4
    """Minimum change to qualify as improvement."""
    
    # =============== GRADIENT ACCUMULATION ===============
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients."""
    
    # =============== MIXED PRECISION ===============
    mixed_precision: bool = True
    """Enable mixed precision training."""
    
    amp_level: str = "O1"  # "O0", "O1", "O2", "O3"
    """Mixed precision optimization level."""
    
    # =============== GRADIENT CHECKPOINTING ===============
    gradient_checkpointing: bool = False
    """Enable gradient checkpointing to save memory."""
    
    # =============== MODEL EMA ===============
    model_ema: bool = True
    """Enable model exponential moving average."""
    
    model_ema_decay: float = 0.9999
    """Decay factor for model EMA."""
    
    model_ema_force_cpu: bool = False
    """Force EMA model to CPU."""
    
    # =============== REGULARIZATION ===============
    weight_decay: float = 1e-4
    """Weight decay for regularization."""
    
    dropout_rate: float = 0.1
    """Dropout rate."""
    
    # =============== WARMUP ===============
    warmup_epochs: int = 5
    """Number of warmup epochs."""
    
    warmup_factor: float = 0.1
    """Initial learning rate multiplier during warmup."""
    
    # =============== METRICS ===============
    metrics: List[str] = field(default_factory=lambda: [
        "loss", "accuracy", "precision", "recall", "mAP"
    ])
    """Metrics to track during training."""
    
    # =============== RESUME TRAINING ===============
    resume_from: Optional[str] = None
    """Path to checkpoint to resume from."""
    
    strict_resume: bool = True
    """Strictly load all weights when resuming."""
    
    # =============== DEBUGGING ===============
    debug: bool = False
    """Enable debug mode."""
    
    overfit_batches: int = 0
    """Overfit on a small number of batches for debugging."""
    
    limit_train_batches: float = 1.0
    """Limit fraction of training batches."""
    
    limit_val_batches: float = 1.0
    """Limit fraction of validation batches."""
    
    fast_dev_run: bool = False
    """Run a quick development loop."""
    
    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()
        self._validate_training_config()
    
    def _validate_training_config(self):
        """Validate training-specific configuration."""
        # Validate sub-configs
        self.dataset.validate()
        self.augmentation.validate()
        self.optimizer.validate()
        self.scheduler.validate()
        self.loss.validate()
        
        # Validate main settings
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        if not 0 <= self.model_ema_decay < 1:
            raise ValueError("model_ema_decay must be between 0 and 1")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        
        if not 0 <= self.warmup_factor <= 1:
            raise ValueError("warmup_factor must be between 0 and 1")
        
        if not 0 <= self.limit_train_batches <= 1:
            raise ValueError("limit_train_batches must be between 0 and 1")
        
        if not 0 <= self.limit_val_batches <= 1:
            raise ValueError("limit_val_batches must be between 0 and 1")
    
    def get_optimizer_params(self, model: torch.nn.Module) -> List[Dict]:
        """Get optimizer parameters with different settings for different parts."""
        params = []
        
        # Separate parameters for backbone, head, etc.
        backbone_params = []
        head_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'head' in name or 'detection' in name:
                head_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for different parts
        if backbone_params:
            params.append({
                'params': backbone_params,
                'lr': self.optimizer.learning_rate * 0.1,  # Lower LR for backbone
                'weight_decay': self.optimizer.weight_decay
            })
        
        if head_params:
            params.append({
                'params': head_params,
                'lr': self.optimizer.learning_rate,
                'weight_decay': self.optimizer.weight_decay
            })
        
        if other_params:
            params.append({
                'params': other_params,
                'lr': self.optimizer.learning_rate,
                'weight_decay': self.optimizer.weight_decay
            })
        
        return params
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration dictionary."""
        config = {
            'type': self.scheduler.type.value,
            't_max': self.scheduler.t_max,
            'eta_min': self.scheduler.eta_min,
            'step_size': self.scheduler.step_size,
            'gamma': self.scheduler.gamma,
            'milestones': self.scheduler.milestones,
            'decay_rate': self.scheduler.decay_rate,
            'patience': self.scheduler.patience,
            'threshold': self.scheduler.threshold,
            'cooldown': self.scheduler.cooldown,
            'min_lr': self.scheduler.min_lr,
            'max_lr': self.scheduler.max_lr,
            'pct_start': self.scheduler.pct_start,
            'div_factor': self.scheduler.div_factor,
            'final_div_factor': self.scheduler.final_div_factor,
            'cycle_mult': self.scheduler.cycle_mult
        }
        
        return config
    
    def display_training_summary(self):
        """Display detailed training configuration summary."""
        super().display()
        
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION DETAILS")
        print("="*60)
        
        # Dataset info
        print(f"\nDataset:")
        print("-" * 40)
        print(f"  Type: {self.dataset.type.value}")
        print(f"  Classes: {self.dataset.num_classes}")
        print(f"  Train split: {self.dataset.train_path}")
        print(f"  Val split: {self.dataset.val_path}")
        
        # Optimization info
        print(f"\nOptimization:")
        print("-" * 40)
        print(f"  Optimizer: {self.optimizer.type.value}")
        print(f"  Learning rate: {self.optimizer.learning_rate}")
        print(f"  Weight decay: {self.optimizer.weight_decay}")
        print(f"  Scheduler: {self.scheduler.type.value}")
        print(f"  Warmup epochs: {self.warmup_epochs}")
        
        # Loss info
        print(f"\nLoss:")
        print("-" * 40)
        print(f"  Type: {self.loss.type.value}")
        print(f"  Box weight: {self.loss.box_loss_weight}")
        print(f"  Obj weight: {self.loss.obj_loss_weight}")
        print(f"  Cls weight: {self.loss.cls_loss_weight}")
        
        # Training schedule
        print(f"\nTraining Schedule:")
        print("-" * 40)
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Checkpoint interval: {self.checkpoint_interval}")
        print(f"  Early stopping patience: {self.early_stopping_patience}")
        
        # Advanced features
        print(f"\nAdvanced Features:")
        print("-" * 40)
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Model EMA: {self.model_ema}")
        print(f"  Distributed: {self.distributed}")
        
        print("="*60)