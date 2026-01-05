# src/training/__init__.py
"""
Training pipeline for Manifold-Constrained Hyper-Connection Vision System.

This module provides:
1. mHC-constrained training with stability guarantees
2. Custom loss functions for hybrid vision tasks
3. Optimizers with manifold constraints
4. Learning rate schedulers with warmup
5. Real-time stability monitoring
"""

from .mhc_trainer import ManifoldConstrainedTrainer
from .loss_functions import (
    MHCYOLOLoss,
    MultiTaskLoss,
    ManifoldRegularizationLoss
)
from .optimizer import (
    ManifoldAwareOptimizer,
    DoublyStochasticProjection
)
from .scheduler import (
    CosineAnnealingWithWarmup,
    PlateauSchedulerWithReset
)
from .stability_monitor import (
    StabilityMonitor,
    TrainingStabilityMetrics
)

__all__ = [
    'ManifoldConstrainedTrainer',
    'MHCYOLOLoss',
    'MultiTaskLoss',
    'ManifoldRegularizationLoss',
    'ManifoldAwareOptimizer',
    'DoublyStochasticProjection',
    'CosineAnnealingWithWarmup',
    'PlateauSchedulerWithReset',
    'StabilityMonitor',
    'TrainingStabilityMetrics'
]