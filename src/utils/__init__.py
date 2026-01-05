# src/utils/__init__.py
"""
Utilities for the Hybrid Vision System.

This package provides:
1. Sinkhorn-Knopp projection for manifold constraints
2. Manifold operations and optimization
3. Comprehensive logging system
4. Evaluation metrics
5. Performance profiling tools
"""

from .sinkhorn import (
    SinkhornKnopp,
    DifferentiableSinkhorn,
    sinkhorn_regularization_loss,
    project_to_doubly_stochastic
)

from .manifold_ops import (
    ManifoldProjector,
    BirkhoffProjector,
    StiefelProjector,
    SPDProjector,
    ManifoldOptimizer,
    manifold_regularization,
    check_manifold_constraints
)

from .logging import (
    StructuredLogger,
    setup_global_logger,
    get_logger,
    log_info,
    log_error,
    log_warning,
    log_debug
)

from .metrics import (
    DetectionMetrics,
    DetectionEvaluator,
    StabilityMetrics,
    InferenceMetrics,
    compute_iou_batch,
    compute_precision_recall,
    smooth_loss_curve
)

from .profiler import (
    ProfileEvent,
    ResourceMonitor,
    ModelProfiler,
    InferenceProfiler,
    get_profiler,
    profile_function,
    measure_memory_usage
)

__all__ = [
    # Sinkhorn
    'SinkhornKnopp',
    'DifferentiableSinkhorn',
    'sinkhorn_regularization_loss',
    'project_to_doubly_stochastic',
    
    # Manifold operations
    'ManifoldProjector',
    'BirkhoffProjector',
    'StiefelProjector',
    'SPDProjector',
    'ManifoldOptimizer',
    'manifold_regularization',
    'check_manifold_constraints',
    
    # Logging
    'StructuredLogger',
    'setup_global_logger',
    'get_logger',
    'log_info',
    'log_error',
    'log_warning',
    'log_debug',
    
    # Metrics
    'DetectionMetrics',
    'DetectionEvaluator',
    'StabilityMetrics',
    'InferenceMetrics',
    'compute_iou_batch',
    'compute_precision_recall',
    'smooth_loss_curve',
    
    # Profiler
    'ProfileEvent',
    'ResourceMonitor',
    'ModelProfiler',
    'InferenceProfiler',
    'get_profiler',
    'profile_function',
    'measure_memory_usage',
]

# Package version
__version__ = "1.0.0"