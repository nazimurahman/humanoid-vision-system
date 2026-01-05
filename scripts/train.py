#!/usr/bin/env python3
"""
Main training script for Hybrid Vision System with Manifold-Constrained Hyper-Connections.
Supports distributed training, mixed precision, and comprehensive logging.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import numpy as np
from pathlib import Path
import random
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from models.hybrid_vision import HybridVisionSystem
from training.mhc_trainer import ManifoldConstrainedTrainer
from data.dataset import COCOVisionDataset
from data.transforms import create_train_transforms
from utils.logging import setup_logging, get_logger
from utils.metrics import TrainingMetricsTracker

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return rank, world_size, local_rank
    return 0, 1, 0  # Single GPU

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_datasets(config, rank, world_size):
    """Create training and validation datasets."""
    
    # Training transforms with augmentation
    train_transforms = create_train_transforms(
        img_size=config.data.image_size,
        augment=config.data.augment,
        mosaic=config.data.use_mosaic,
        mixup=config.data.use_mixup,
        hsv_prob=config.data.hsv_prob
    )
    
    # Validation transforms (no augmentation)
    val_transforms = create_train_transforms(
        img_size=config.data.image_size,
        augment=False,
        mosaic=False,
        mixup=False
    )
    
    # Create datasets
    train_dataset = COCOVisionDataset(
        root=config.data.train_path,
        annotation_file=config.data.train_annotations,
        transforms=train_transforms,
        cache=config.data.cache_images
    )
    
    val_dataset = COCOVisionDataset(
        root=config.data.val_path,
        annotation_file=config.data.val_annotations,
        transforms=val_transforms,
        cache=config.data.cache_images
    )
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    
    return train_dataset, val_dataset, train_sampler, val_sampler

def create_dataloaders(train_dataset, val_dataset, train_sampler, val_sampler, config, rank):
    """Create data loaders."""
    
    # Training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
    )
    
    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None
    )
    
    return train_loader, val_loader

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Merge with defaults
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    return {
        'model': model_config,
        'training': training_config,
        'data': config_dict.get('data', {}),
        'logging': config_dict.get('logging', {})
    }

def main(args):
    """Main training function."""
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set random seed
    set_seed(args.seed + rank)
    
    # Setup logging
    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir / 'training.log', rank=rank)
    logger = get_logger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb (only on rank 0)
    if rank == 0 and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f'hybrid-vision-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=config
        )
    
    # Create model
    logger.info(f"Creating model on rank {rank}")
    model = HybridVisionSystem(
        config=config['model'],
        num_classes=config['data'].get('num_classes', 80),
        use_vit=config['model'].use_vit,
        use_rag=config['model'].use_rag
    )
    
    # Move to device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config['training'].find_unused_params
        )
    
    # Create datasets and loaders
    logger.info(f"Creating datasets on rank {rank}")
    train_dataset, val_dataset, train_sampler, val_sampler = create_datasets(
        config, rank, world_size
    )
    
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, train_sampler, val_sampler,
        config, rank
    )
    
    # Create trainer
    trainer = ManifoldConstrainedTrainer(
        model=model.module if world_size > 1 else model,
        config=config['training'],
        device=device
    )
    
    # Create metrics tracker
    metrics_tracker = TrainingMetricsTracker(
        log_dir=log_dir,
        save_interval=config['logging'].get('save_interval', 100)
    )
    
    # Training loop
    logger.info(f"Starting training on rank {rank}")
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            log_wandb=args.wandb and rank == 0
        )
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if world_size > 1:
            dist.destroy_process_group()
        
        if rank == 0 and args.wandb:
            wandb.finish()
        
        logger.info("Training completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hybrid Vision System')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Logging and monitoring
    parser.add_argument('--log-dir', type=str, default='logs/training',
                       help='Directory for logs and checkpoints')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='humanoid-vision',
                       help='W&B project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='W&B run name')
    
    # Distributed training
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    # Debugging
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (small dataset)')
    
    args = parser.parse_args()
    
    # Set environment variables for distributed training
    if args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    main(args)