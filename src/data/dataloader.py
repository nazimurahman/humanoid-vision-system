# src/data/dataloader.py

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any, List, Callable, Union
import multiprocessing as mp
from queue import Queue
from threading import Thread
import time
import logging

logger = logging.getLogger(__name__)

class MHCDataLoader(DataLoader):
    """
    Enhanced DataLoader for mHC training with:
    - Automatic mixed precision support
    - Gradient accumulation aware batching
    - Distributed training optimization
    - Memory-efficient prefetching
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        timeout: int = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context: str = 'spawn',
        gradient_accumulation_steps: int = 1,
        use_mixed_precision: bool = True
    ):
        """
        Initialize enhanced DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            collate_fn: Custom collate function
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Number of batches prefetched by each worker
            timeout: Timeout for collecting batches from workers
            worker_init_fn: Function to initialize workers
            multiprocessing_context: Multiprocessing context
            gradient_accumulation_steps: For gradient accumulation compatibility
            use_mixed_precision: Optimize for mixed precision training
        """
        # Use dataset's collate_fn if none provided
        if collate_fn is None and hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        
        # Calculate effective batch size for gradient accumulation
        effective_batch_size = batch_size // gradient_accumulation_steps
        if effective_batch_size < 1:
            effective_batch_size = 1
        
        # Initialize parent DataLoader
        super().__init__(
            dataset=dataset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context
        )
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        
        # Statistics
        self.batches_loaded = 0
        self.total_samples = 0
        
        logger.info(f"DataLoader initialized with {num_workers} workers")
        logger.info(f"Batch size: {batch_size} (effective: {effective_batch_size})")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        
    def __iter__(self):
        """Custom iterator with gradient accumulation support."""
        self.batches_loaded = 0
        self.total_samples = 0
        
        if self.gradient_accumulation_steps > 1:
            return self._accumulation_iterator()
        else:
            return super().__iter__()
    
    def _accumulation_iterator(self):
        """Iterator that accumulates batches for gradient accumulation."""
        accumulated_batches = []
        accumulated_count = 0
        
        for batch in super().__iter__():
            accumulated_batches.append(batch)
            accumulated_count += 1
            
            if accumulated_count == self.gradient_accumulation_steps:
                # Merge accumulated batches
                merged_batch = self._merge_batches(accumulated_batches)
                yield merged_batch
                
                # Reset accumulation
                accumulated_batches = []
                accumulated_count = 0
        
        # Handle remaining batches
        if accumulated_batches:
            merged_batch = self._merge_batches(accumulated_batches)
            yield merged_batch
    
    def _merge_batches(self, batches: List[Dict]) -> Dict:
        """Merge multiple batches into one."""
        if not batches:
            return {}
        
        # Initialize merged batch
        merged = {}
        
        # Merge each key
        for key in batches[0].keys():
            if key == 'images':
                # Stack images
                images = torch.cat([batch[key] for batch in batches], dim=0)
                merged[key] = images
            elif key == 'targets':
                # Merge targets dictionary
                merged_targets = {}
                for target_key in batches[0][key].keys():
                    if target_key == 'boxes':
                        # Concatenate boxes
                        boxes_list = [batch[key][target_key] for batch in batches]
                        merged_targets[target_key] = torch.cat(boxes_list, dim=0)
                    elif target_key == 'labels':
                        # Concatenate labels
                        labels_list = [batch[key][target_key] for batch in batches]
                        merged_targets[target_key] = torch.cat(labels_list, dim=0)
                    elif target_key == 'area':
                        # Concatenate areas
                        area_list = [batch[key][target_key] for batch in batches]
                        merged_targets[target_key] = torch.cat(area_list, dim=0)
                    elif target_key == 'box_mask':
                        # Concatenate box masks
                        mask_list = [batch[key][target_key] for batch in batches]
                        merged_targets[target_key] = torch.cat(mask_list, dim=0)
                merged[key] = merged_targets
            elif key in ['image_ids', 'original_sizes']:
                # Concatenate IDs and sizes
                merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
        
        return merged
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DataLoader statistics."""
        return {
            'batches_loaded': self.batches_loaded,
            'total_samples': self.total_samples,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'effective_batch_size': self.batch_size * self.gradient_accumulation_steps
        }

class StreamingDataLoader:
    """
    DataLoader for streaming data from cameras or network sources.
    
    Features:
    - Real-time streaming support
    - Frame skipping for variable FPS
    - Buffer management
    - Hardware acceleration
    """
    
    def __init__(
        self,
        stream_source: Union[str, int],  # URL or camera index
        batch_size: int = 1,
        frame_size: tuple = (416, 416),
        fps: int = 30,
        buffer_size: int = 10,
        transform: Optional[Callable] = None
    ):
        """
        Initialize streaming DataLoader.
        
        Args:
            stream_source: Camera index (0 for webcam) or RTSP/HTTP URL
            batch_size: Batch size (usually 1 for streaming)
            frame_size: Target frame size (height, width)
            fps: Target frames per second
            buffer_size: Size of frame buffer
            transform: Transformations to apply to frames
        """
        import cv2
        
        self.stream_source = stream_source
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.target_fps = fps
        self.buffer_size = buffer_size
        self.transform = transform
        
        # Open video stream
        self.cap = cv2.VideoCapture(stream_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open stream source: {stream_source}")
        
        # Get actual stream properties
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate skip frames for target FPS
        self.skip_frames = max(1, int(self.actual_fps / self.target_fps))
        
        # Frame buffer
        self.buffer = Queue(maxsize=buffer_size)
        
        # Start frame reader thread
        self.reader_thread = Thread(target=self._frame_reader, daemon=True)
        self.reader_thread.start()
        
        # Statistics
        self.frames_read = 0
        self.frames_processed = 0
        
        logger.info(f"Streaming from {stream_source}")
        logger.info(f"Resolution: {self.frame_width}x{self.frame_height}")
        logger.info(f"FPS: {self.actual_fps} (target: {self.target_fps}, skip: {self.skip_frames})")
    
    def _frame_reader(self):
        """Read frames from stream in background thread."""
        import cv2
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from stream")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Skip frames to achieve target FPS
            if frame_count % self.skip_frames != 0:
                continue
            
            # Resize frame
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Put frame in buffer (block if full)
            self.buffer.put(frame)
            self.frames_read += 1
    
    def __iter__(self):
        """Iterator for streaming frames."""
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        """Get next batch of frames."""
        frames = []
        
        for _ in range(self.batch_size):
            # Get frame from buffer (block if empty)
            frame = self.buffer.get()
            
            # Apply transforms if provided
            if self.transform:
                transformed = self.transform(image=frame)
                frame = transformed['image']
            else:
                # Convert to tensor
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # HWC to CHW
            
            frames.append(frame)
        
        self.frames_processed += len(frames)
        
        # Stack frames if batch_size > 1
        if self.batch_size > 1:
            frames = torch.stack(frames)
        else:
            frames = frames[0].unsqueeze(0)  # Add batch dimension
        
        return {'images': frames}
    
    def __len__(self):
        """Return infinity for streaming (continuous)."""
        return float('inf')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            'frames_read': self.frames_read,
            'frames_processed': self.frames_processed,
            'buffer_size': self.buffer.qsize(),
            'actual_fps': self.actual_fps,
            'target_fps': self.target_fps
        }
    
    def close(self):
        """Close stream and cleanup."""
        if hasattr(self, 'cap'):
            self.cap.release()
        logger.info("Stream closed")

class DistributedDataLoaderWrapper:
    """
    Wrapper for distributed training with mHC.
    
    Handles:
    - Automatic world size detection
    - Rank-based sampling
    - Gradient synchronization
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42
    ):
        self.dataloader = dataloader
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        
        # Create distributed sampler if needed
        if world_size > 1:
            self.sampler = DistributedSampler(
                dataset=dataloader.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=dataloader.shuffle,
                seed=seed
            )
            dataloader.sampler = self.sampler
            dataloader.shuffle = False
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        if hasattr(self, 'sampler'):
            self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def __getattr__(self, name):
        """Delegate other attributes to underlying dataloader."""
        return getattr(self.dataloader, name)