# src/inference/preprocessing.py
"""
Image Preprocessing Pipeline for Robotic Vision

Features:
1. Real-time image preprocessing optimized for mHC models
2. Camera streaming and management
3. Multi-camera support
4. Frame synchronization
5. Image augmentation for inference
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import time
import threading
from queue import Queue, LifoQueue
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress OpenCV warnings
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    # Image dimensions
    target_height: int = 416
    target_width: int = 416
    maintain_aspect_ratio: bool = False
    padding_mode: str = "constant"  # constant, edge, reflect
    
    # Color processing
    rgb_input: bool = True
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation (for inference-time augmentation)
    enable_augmentation: bool = False
    augment_probability: float = 0.5
    augment_brightness: float = 0.2
    augment_contrast: float = 0.2
    augment_saturation: float = 0.2
    augment_hue: float = 0.1
    
    # Performance
    use_gpu: bool = True
    half_precision: bool = True
    cache_size: int = 10  # Number of processed images to cache

class PreprocessingMode(Enum):
    """Different preprocessing modes."""
    FAST = "fast"        # Minimal processing for real-time
    ACCURATE = "accurate" # Full processing for accuracy
    AUGMENTED = "augmented" # With inference-time augmentation
    EDGE = "edge"        # Optimized for edge devices

class ImagePreprocessor:
    """
    High-performance image preprocessor for robotic vision.
    
    Optimized for:
    1. Real-time processing (<5ms per frame)
    2. GPU acceleration when available
    3. Memory efficiency
    4. Consistency with training preprocessing
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize image preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build preprocessing pipeline
        self._build_pipeline()
        
        # Cache for processed images
        self.cache = {}
        self.cache_keys = []
        
        logger.info(f"ImagePreprocessor initialized (mode: {self.config.target_height}x{self.config.target_width})")
        logger.info(f"Device: {self.device}, Half precision: {self.config.half_precision}")
    
    def _build_pipeline(self):
        """Build preprocessing pipeline based on configuration."""
        pipeline = []
        
        # Resize transformation
        if self.config.maintain_aspect_ratio:
            pipeline.append(transforms.Resize(self.config.target_height))
            pipeline.append(transforms.CenterCrop((self.config.target_height, self.config.target_width)))
        else:
            pipeline.append(transforms.Resize((self.config.target_height, self.config.target_width)))
        
        # Color augmentation (if enabled)
        if self.config.enable_augmentation:
            pipeline.append(transforms.ColorJitter(
                brightness=self.config.augment_brightness,
                contrast=self.config.augment_contrast,
                saturation=self.config.augment_saturation,
                hue=self.config.augment_hue,
            ))
        
        # Convert to tensor
        pipeline.append(transforms.ToTensor())
        
        # Normalization (if enabled)
        if self.config.normalize:
            pipeline.append(transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std
            ))
        
        # Create composed transform
        self.transform = transforms.Compose(pipeline)
        
        # GPU-accelerated transforms if available
        if self.config.use_gpu and torch.cuda.is_available():
            self._build_gpu_pipeline()
    
    def _build_gpu_pipeline(self):
        """Build GPU-accelerated preprocessing pipeline."""
        try:
            import kornia
            from kornia import augmentation as K
            
            self.use_kornia = True
            
            # Build Kornia augmentation pipeline
            gpu_pipeline = []
            
            # Resize
            gpu_pipeline.append(K.Resize(
                size=(self.config.target_height, self.config.target_width),
                interpolation='bilinear',
                align_corners=False
            ))
            
            # Color augmentation
            if self.config.enable_augmentation:
                gpu_pipeline.append(K.ColorJitter(
                    brightness=self.config.augment_brightness,
                    contrast=self.config.augment_contrast,
                    saturation=self.config.augment_saturation,
                    hue=self.config.augment_hue,
                    p=self.config.augment_probability
                ))
            
            # Normalization
            if self.config.normalize:
                gpu_pipeline.append(K.Normalize(
                    mean=torch.tensor(self.config.mean).view(1, 3, 1, 1),
                    std=torch.tensor(self.config.std).view(1, 3, 1, 1)
                ))
            
            self.gpu_transform = torch.nn.Sequential(*gpu_pipeline)
            self.gpu_transform = self.gpu_transform.to(self.device)
            
            logger.info("GPU-accelerated preprocessing pipeline built")
            
        except ImportError:
            self.use_kornia = False
            logger.warning("Kornia not available, using CPU preprocessing")
    
    def process(self, image: np.ndarray, mode: PreprocessingMode = PreprocessingMode.FAST) -> torch.Tensor:
        """
        Process single image for inference.
        
        Args:
            image: Input image (BGR or RGB)
            mode: Preprocessing mode
            
        Returns:
            Preprocessed tensor [C, H, W]
        """
        # Generate cache key
        cache_key = self._get_cache_key(image, mode)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Convert color space if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                if self.config.rgb_input and image.shape[2] == 3:
                    # BGR to RGB conversion
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                # Handle grayscale
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Handle grayscale (2D)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Choose processing method based on mode and availability
        if mode == PreprocessingMode.FAST and self.use_kornia:
            tensor = self._process_gpu_fast(image_rgb)
        elif mode == PreprocessingMode.ACCURATE and self.use_kornia:
            tensor = self._process_gpu_accurate(image_rgb)
        else:
            tensor = self._process_cpu(image_rgb)
        
        # Convert to half precision if enabled
        if self.config.half_precision:
            tensor = tensor.half()
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Cache result
        self._add_to_cache(cache_key, tensor)
        
        return tensor
    
    def _process_gpu_fast(self, image: np.ndarray) -> torch.Tensor:
        """Fast GPU processing using Kornia."""
        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Apply GPU transforms
        with torch.no_grad():
            tensor = self.gpu_transform(tensor)
        
        return tensor.squeeze(0)
    
    def _process_gpu_accurate(self, image: np.ndarray) -> torch.Tensor:
        """Accurate GPU processing with proper interpolation."""
        # Use OpenCV for high-quality resize
        resized = cv2.resize(
            image,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert to tensor
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Apply only normalization on GPU
        if self.config.normalize:
            mean = torch.tensor(self.config.mean).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor(self.config.std).view(1, 3, 1, 1).to(self.device)
            tensor = (tensor - mean) / std
        
        return tensor.squeeze(0)
    
    def _process_cpu(self, image: np.ndarray) -> torch.Tensor:
        """CPU processing using torchvision."""
        # Convert to PIL Image for torchvision
        from PIL import Image
        pil_image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor
    
    def _get_cache_key(self, image: np.ndarray, mode: PreprocessingMode) -> str:
        """Generate cache key for image."""
        # Simple hash based on image shape and mode
        return f"{image.shape}_{image.dtype}_{mode.value}"
    
    def _add_to_cache(self, key: str, tensor: torch.Tensor):
        """Add processed tensor to cache."""
        self.cache[key] = tensor
        self.cache_keys.append(key)
        
        # Limit cache size
        if len(self.cache) > self.config.cache_size:
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
    
    def clear_cache(self):
        """Clear preprocessing cache."""
        self.cache.clear()
        self.cache_keys.clear()
        logger.info("Preprocessing cache cleared")
    
    def batch_process(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Process batch of images efficiently.
        
        Args:
            images: List of input images
            
        Returns:
            Batch tensor [B, C, H, W]
        """
        processed = []
        
        for img in images:
            tensor = self.process(img)
            processed.append(tensor)
        
        return torch.stack(processed)
    
    def create_attention_mask(self, image: np.ndarray, 
                            roi: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        """
        Create attention mask for region of interest.
        
        Args:
            image: Input image
            roi: Optional region of interest (x1, y1, x2, y2)
            
        Returns:
            Attention mask tensor
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        
        if roi is not None:
            x1, y1, x2, y2 = roi
            mask[y1:y2, x1:x2] = 1.0
        else:
            # Create center-weighted mask
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            mask = 1.0 - distance / max_distance
        
        # Resize to target size
        mask_resized = cv2.resize(
            mask,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask_resized).float()
        mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
        mask_tensor = mask_tensor.to(self.device)
        
        return mask_tensor

class VideoStreamer:
    """
    Real-time video streamer for robotic cameras.
    
    Features:
    1. Multi-camera support
    2. Frame synchronization
    3. Buffer management
    4. Error recovery
    """
    
    def __init__(self, camera_ids: Union[int, List[int]], config: Optional[Dict] = None):
        """
        Initialize video streamer.
        
        Args:
            camera_ids: Single camera ID or list of camera IDs
            config: Streamer configuration
        """
        self.camera_ids = [camera_ids] if isinstance(camera_ids, int) else camera_ids
        self.config = config or {}
        
        # Camera properties
        self.frame_width = self.config.get('frame_width', 640)
        self.frame_height = self.config.get('frame_height', 480)
        self.fps = self.config.get('fps', 30)
        self.buffer_size = self.config.get('buffer_size', 3)
        
        # Camera instances
        self.cameras = {}
        self.frame_buffers = {}
        self.latest_frames = {}
        self.running = False
        
        # Thread management
        self.camera_threads = []
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.frame_counters = {}
        self.frame_times = {}
        
        logger.info(f"VideoStreamer initialized for cameras: {self.camera_ids}")
    
    def start(self):
        """Start all camera streams."""
        if self.running:
            logger.warning("Streamer already running")
            return
        
        self.running = True
        
        for cam_id in self.camera_ids:
            # Initialize camera
            cap = cv2.VideoCapture(cam_id)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {cam_id}")
                continue
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Create buffer
            self.frame_buffers[cam_id] = Queue(maxsize=self.buffer_size)
            self.latest_frames[cam_id] = None
            self.frame_counters[cam_id] = 0
            self.frame_times[cam_id] = deque(maxlen=100)
            
            # Start camera thread
            thread = threading.Thread(
                target=self._camera_loop,
                args=(cam_id, cap),
                daemon=True
            )
            thread.start()
            self.camera_threads.append(thread)
            
            self.cameras[cam_id] = cap
            logger.info(f"Camera {cam_id} started")
        
        if not self.cameras:
            raise RuntimeError("No cameras could be opened")
        
        logger.info(f"VideoStreamer started with {len(self.cameras)} cameras")
    
    def stop(self):
        """Stop all camera streams."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.camera_threads:
            thread.join(timeout=2.0)
        
        # Release cameras
        for cam_id, cap in self.cameras.items():
            cap.release()
            logger.info(f"Camera {cam_id} released")
        
        self.cameras.clear()
        self.camera_threads.clear()
        logger.info("VideoStreamer stopped")
    
    def _camera_loop(self, cam_id: int, cap: cv2.VideoCapture):
        """Camera capture loop."""
        logger.info(f"Camera loop started for camera {cam_id}")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    logger.error(f"Failed to read frame from camera {cam_id}")
                    time.sleep(0.01)
                    continue
                
                # Update frame time
                frame_time = time.time() - start_time
                self.frame_times[cam_id].append(frame_time)
                
                # Update frame counter
                self.frame_counters[cam_id] += 1
                
                with self.lock:
                    self.latest_frames[cam_id] = frame
                
                # Add to buffer (non-blocking)
                try:
                    self.frame_buffers[cam_id].put_nowait(frame)
                except queue.Full:
                    # Discard oldest frame
                    try:
                        self.frame_buffers[cam_id].get_nowait()
                        self.frame_buffers[cam_id].put_nowait(frame)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Error in camera loop {cam_id}: {e}")
                time.sleep(0.1)
        
        logger.info(f"Camera loop stopped for camera {cam_id}")
    
    def get_frame(self, cam_id: int, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get latest frame from camera.
        
        Args:
            cam_id: Camera ID
            timeout: Timeout in seconds
            
        Returns:
            Latest frame or None
        """
        if cam_id not in self.cameras:
            logger.error(f"Camera {cam_id} not found")
            return None
        
        # Try to get from buffer first
        try:
            frame = self.frame_buffers[cam_id].get(timeout=timeout)
            return frame
        except queue.Empty:
            # Fall back to latest frame
            with self.lock:
                return self.latest_frames.get(cam_id)
    
    def get_synchronized_frames(self, timeout: float = 0.5) -> Dict[int, np.ndarray]:
        """
        Get synchronized frames from all cameras.
        
        Args:
            timeout: Maximum time to wait for synchronization
            
        Returns:
            Dictionary of camera_id -> frame
        """
        frames = {}
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_available = True
            frames.clear()
            
            for cam_id in self.cameras.keys():
                frame = self.get_frame(cam_id, timeout=0.01)
                if frame is None:
                    all_available = False
                    break
                frames[cam_id] = frame
            
            if all_available and frames:
                return frames
        
        logger.warning("Failed to synchronize frames within timeout")
        return frames
    
    def get_performance_stats(self, cam_id: int) -> Dict[str, Any]:
        """
        Get performance statistics for camera.
        
        Args:
            cam_id: Camera ID
            
        Returns:
            Performance statistics
        """
        if cam_id not in self.cameras:
            return {}
        
        times = list(self.frame_times[cam_id])
        stats = {
            'camera_id': cam_id,
            'frame_count': self.frame_counters[cam_id],
            'buffer_size': self.frame_buffers[cam_id].qsize(),
            'frame_time_ms': {
                'mean': np.mean(times) * 1000 if times else 0,
                'std': np.std(times) * 1000 if times else 0,
                'min': np.min(times) * 1000 if times else 0,
                'max': np.max(times) * 1000 if times else 0,
            } if times else {},
            'effective_fps': 1.0 / np.mean(times) if times else 0,
            'is_capturing': self.cameras[cam_id].isOpened(),
        }
        
        return stats

class CameraManager:
    """
    Manages multiple camera streams with advanced features.
    
    Features:
    1. Camera discovery and management
    2. Automatic reconnection
    3. Calibration support
    4. Frame alignment
    """
    
    def __init__(self):
        """Initialize camera manager."""
        self.cameras = {}
        self.streamers = {}
        self.calibrations = {}
        self.camera_profiles = {}
        
        # Thread pool for camera operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Camera discovery
        self.available_cameras = self._discover_cameras()
        
        logger.info(f"CameraManager initialized. Found {len(self.available_cameras)} cameras")
    
    def _discover_cameras(self) -> List[int]:
        """Discover available cameras."""
        available = []
        
        # Try first 10 camera indices
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                available.append(i)
                cap.release()
            
            # Small delay to avoid overwhelming system
            time.sleep(0.01)
        
        return available
    
    def add_camera(self, cam_id: int, profile: Optional[Dict] = None):
        """
        Add camera with specific profile.
        
        Args:
            cam_id: Camera ID
            profile: Camera profile configuration
        """
        if cam_id not in self.available_cameras:
            logger.warning(f"Camera {cam_id} not available")
            return
        
        profile = profile or {
            'name': f'camera_{cam_id}',
            'width': 640,
            'height': 480,
            'fps': 30,
            'color_mode': 'bgr',
            'auto_exposure': True,
            'auto_white_balance': True,
        }
        
        # Create streamer for camera
        streamer = VideoStreamer(cam_id, config=profile)
        
        # Store references
        self.cameras[cam_id] = streamer
        self.camera_profiles[cam_id] = profile
        
        logger.info(f"Camera {cam_id} added with profile: {profile}")
    
    def start_all(self):
        """Start all cameras."""
        for cam_id, streamer in self.cameras.items():
            try:
                streamer.start()
                logger.info(f"Camera {cam_id} started")
            except Exception as e:
                logger.error(f"Failed to start camera {cam_id}: {e}")
    
    def stop_all(self):
        """Stop all cameras."""
        for cam_id, streamer in self.cameras.items():
            try:
                streamer.stop()
                logger.info(f"Camera {cam_id} stopped")
            except Exception as e:
                logger.error(f"Failed to stop camera {cam_id}: {e}")
    
    def get_camera_frame(self, cam_id: int, preprocess: bool = False, 
                        preprocessor: Optional[ImagePreprocessor] = None) -> Union[np.ndarray, torch.Tensor]:
        """
        Get frame from camera with optional preprocessing.
        
        Args:
            cam_id: Camera ID
            preprocess: Whether to preprocess the frame
            preprocessor: Optional preprocessor instance
            
        Returns:
            Frame (numpy array or tensor)
        """
        if cam_id not in self.cameras:
            raise ValueError(f"Camera {cam_id} not managed")
        
        streamer = self.cameras[cam_id]
        frame = streamer.get_frame(cam_id)
        
        if frame is None:
            raise RuntimeError(f"Failed to get frame from camera {cam_id}")
        
        if preprocess:
            if preprocessor is None:
                preprocessor = ImagePreprocessor()
            return preprocessor.process(frame)
        
        return frame
    
    def calibrate_camera(self, cam_id: int, pattern_size: Tuple[int, int] = (9, 6),
                        square_size: float = 0.025) -> bool:
        """
        Calibrate camera using chessboard pattern.
        
        Args:
            cam_id: Camera ID
            pattern_size: Chessboard pattern size (corners)
            square_size: Size of squares in meters
            
        Returns:
            True if calibration successful
        """
        logger.info(f"Starting calibration for camera {cam_id}")
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        streamer = self.cameras[cam_id]
        frames_captured = 0
        max_frames = 20
        
        while frames_captured < max_frames:
            frame = streamer.get_frame(cam_id)
            if frame is None:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                
                # Draw and display corners
                frame_with_corners = frame.copy()
                cv2.drawChessboardCorners(frame_with_corners, pattern_size, corners_refined, ret)
                cv2.imshow(f'Camera {cam_id} Calibration', frame_with_corners)
                cv2.waitKey(500)
                
                frames_captured += 1
                logger.info(f"Calibration frame {frames_captured}/{max_frames} captured")
            
            time.sleep(0.1)
        
        cv2.destroyAllWindows()
        
        if len(objpoints) < 5:
            logger.error(f"Insufficient calibration frames: {len(objpoints)}")
            return False
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if not ret:
            logger.error("Camera calibration failed")
            return False
        
        # Save calibration
        self.calibrations[cam_id] = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'ret': ret,
            'image_size': gray.shape[::-1]
        }
        
        logger.info(f"Camera {cam_id} calibration successful")
        logger.info(f"Camera matrix:\n{camera_matrix}")
        logger.info(f"Distortion coefficients: {dist_coeffs.ravel()}")
        
        return True
    
    def undistort_frame(self, cam_id: int, frame: np.ndarray) -> np.ndarray:
        """
        Undistort frame using camera calibration.
        
        Args:
            cam_id: Camera ID
            frame: Input frame
            
        Returns:
            Undistorted frame
        """
        if cam_id not in self.calibrations:
            logger.warning(f"No calibration found for camera {cam_id}")
            return frame
        
        calib = self.calibrations[cam_id]
        h, w = frame.shape[:2]
        
        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            calib['camera_matrix'],
            calib['dist_coeffs'],
            (w, h),
            1,
            (w, h)
        )
        
        # Undistort
        undistorted = cv2.undistort(
            frame,
            calib['camera_matrix'],
            calib['dist_coeffs'],
            None,
            new_camera_matrix
        )
        
        # Crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def get_all_frames(self, preprocess: bool = False) -> Dict[int, Union[np.ndarray, torch.Tensor]]:
        """
        Get frames from all cameras.
        
        Args:
            preprocess: Whether to preprocess frames
            
        Returns:
            Dictionary of camera_id -> frame
        """
        frames = {}
        
        for cam_id, streamer in self.cameras.items():
            try:
                if preprocess:
                    frames[cam_id] = self.get_camera_frame(cam_id, preprocess=True)
                else:
                    frame = streamer.get_frame(cam_id, timeout=0.1)
                    if frame is not None:
                        frames[cam_id] = frame
            except Exception as e:
                logger.error(f"Failed to get frame from camera {cam_id}: {e}")
        
        return frames