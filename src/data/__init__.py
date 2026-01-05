# src/data/__init__.py

from .dataset import BaseVisionDataset
from .transforms import (
    MHCTransformComposer,
    MosaicAugmentation,
    MixupAugmentation,
    RandomErasing,
    GPUAcceleratedTransforms,
    AdaptiveAugmentation
)
from .dataloader import (
    MHCDataLoader,
    StreamingDataLoader,
    DistributedDataLoaderWrapper
)
from .coco import (
    COCODataset,
    COCODataModule
)
from .streaming import (
    RoboticCameraStream,
    MultiCameraManager,
    StreamConfig,
    StreamType
)

__all__ = [
    # Dataset
    'BaseVisionDataset',
    
    # Transforms
    'MHCTransformComposer',
    'MosaicAugmentation',
    'MixupAugmentation',
    'RandomErasing',
    'GPUAcceleratedTransforms',
    'AdaptiveAugmentation',
    
    # DataLoader
    'MHCDataLoader',
    'StreamingDataLoader',
    'DistributedDataLoaderWrapper',
    
    # COCO
    'COCODataset',
    'COCODataModule',
    
    # Streaming
    'RoboticCameraStream',
    'MultiCameraManager',
    'StreamConfig',
    'StreamType'
]

# Version
__version__ = "1.0.0"