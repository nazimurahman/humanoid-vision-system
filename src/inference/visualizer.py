# src/inference/visualizer.py
"""
Visualization tools for robotic vision inference.

Features:
1. Real-time detection visualization
2. Performance monitoring and display
3. Debug visualization for mHC layers
4. Robot state visualization
5. Multi-camera view support
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
import time
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import deque
import logging
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    # Display settings
    display_width: int = 1280
    display_height: int = 720
    show_fps: bool = True
    show_tracking: bool = True
    show_class_labels: bool = True
    show_confidence: bool = True
    
    # Color settings
    color_palette: str = "hsv"  # hsv, tab20, viridis, rainbow
    box_thickness: int = 2
    text_thickness: int = 1
    text_scale: float = 0.5
    
    # Performance display
    show_performance_stats: bool = True
    performance_window_size: int = 100
    plot_colors: List[str] = None
    
    # Debug visualization
    show_mhc_activations: bool = False
    show_attention_maps: bool = False
    show_feature_maps: bool = False
    
    # Robot visualization
    show_robot_state: bool = True
    robot_state_update_rate: float = 10.0  # Hz
    
    def __post_init__(self):
        if self.plot_colors is None:
            self.plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

class VisualizationMode(Enum):
    """Visualization modes."""
    DETECTION = "detection"
    PERFORMANCE = "performance"
    DEBUG = "debug"
    ROBOT = "robot"
    COMBINED = "combined"

class DetectionVisualizer:
    """
    Real-time detection visualizer for robotic vision.
    
    Features:
    1. Multi-object visualization with tracking
    2. Custom color palettes
    3. Configurable display options
    4. Performance overlay
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize detection visualizer.
        
        Args:
            config: Visualization configuration
            class_names: List of class names for labeling
        """
        self.config = config or VisualizationConfig()
        self.class_names = class_names or self._get_default_class_names()
        
        # Initialize color palette
        self.color_palette = self._create_color_palette(len(self.class_names))
        
        # Performance tracking
        self.fps_history = deque(maxlen=self.config.performance_window_size)
        self.inference_times = deque(maxlen=self.config.performance_window_size)
        self.last_update_time = time.time()
        self.frame_counter = 0
        
        # Font for text (try to load system font)
        self.font = self._load_font()
        
        # Create colormap for debug visualizations
        self.debug_cmap = cm.get_cmap('viridis')
        
        logger.info("DetectionVisualizer initialized")
    
    def _get_default_class_names(self) -> List[str]:
        """Get default COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def _create_color_palette(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Create color palette for different classes."""
        if self.config.color_palette == "hsv":
            # HSV color palette (good for many classes)
            colors = []
            for i in range(num_classes):
                hue = i / max(num_classes, 1)
                # Convert HSV to BGR (OpenCV uses BGR)
                color_hsv = np.uint8([[[hue * 179, 255, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                colors.append(tuple(map(int, color_bgr)))
            return colors
        
        elif self.config.color_palette == "tab20":
            # Matplotlib tab20 colormap
            cmap = cm.get_cmap('tab20')
            colors = []
            for i in range(num_classes):
                rgba = cmap(i % 20)
                # Convert RGBA to BGR
                bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
                colors.append(bgr)
            return colors
        
        else:
            # Default: distinct colors
            distinct_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (128, 0, 0),    # Maroon
                (0, 128, 0),    # Dark Green
                (0, 0, 128),    # Navy
                (128, 128, 0),  # Olive
                (128, 0, 128),  # Purple
                (0, 128, 128),  # Teal
            ]
            
            # Cycle through distinct colors if we have more classes
            colors = []
            for i in range(num_classes):
                colors.append(distinct_colors[i % len(distinct_colors)])
            return colors
    
    def _load_font(self):
        """Load font for text rendering."""
        # Try to load system fonts
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            'arial.ttf',
        ]
        
        for font_path in font_paths:
            try:
                # PIL font for better quality
                font = ImageFont.truetype(font_path, 
                                         int(16 * self.config.text_scale))
                logger.info(f"Loaded font: {font_path}")
                return font
            except IOError:
                continue
        
        logger.warning("Could not load system font, using default")
        return ImageFont.load_default()
    
    def visualize_detections(self, image: np.ndarray,
                            detections: Dict[str, Any],
                            mode: VisualizationMode = VisualizationMode.DETECTION) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image (BGR format)
            detections: Detection results
            mode: Visualization mode
            
        Returns:
            Image with visualizations
        """
        # Make a copy to avoid modifying original
        vis_image = image.copy()
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Apply different visualizations based on mode
        if mode == VisualizationMode.DETECTION:
            vis_image = self._draw_detections(vis_image, detections)
        elif mode == VisualizationMode.PERFORMANCE:
            vis_image = self._draw_performance_overlay(vis_image, detections)
        elif mode == VisualizationMode.DEBUG:
            vis_image = self._draw_debug_info(vis_image, detections)
        elif mode == VisualizationMode.ROBOT:
            vis_image = self._draw_robot_state(vis_image, detections)
        elif mode == VisualizationMode.COMBINED:
            vis_image = self._draw_detections(vis_image, detections)
            vis_image = self._draw_performance_overlay(vis_image, detections)
            if 'debug_info' in detections:
                vis_image = self._draw_debug_info(vis_image, detections)
        
        # Resize if needed
        if vis_image.shape[1] != self.config.display_width or \
           vis_image.shape[0] != self.config.display_height:
            vis_image = cv2.resize(
                vis_image,
                (self.config.display_width, self.config.display_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        return vis_image
    
    def _draw_detections(self, image: np.ndarray, 
                        detections: Dict[str, Any]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        if 'detections' not in detections:
            return image
        
        dets = detections['detections']
        boxes = dets.get('boxes', [])
        scores = dets.get('scores', [])
        class_ids = dets.get('class_ids', [])
        track_ids = dets.get('track_ids', [])
        
        if not boxes:
            return image
        
        # Convert image to PIL for better text rendering
        if self.config.show_class_labels or self.config.show_confidence:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            # Skip if confidence is too low
            if score < 0.1:  # Minimum confidence for visualization
                continue
            
            # Get color for this class/track
            if self.config.show_tracking and i < len(track_ids):
                color_idx = track_ids[i] % len(self.color_palette)
            else:
                color_idx = class_id % len(self.color_palette)
            
            color = self.color_palette[color_idx]
            
            # Convert box from center format to corner format
            if len(box) == 4:  # [x_center, y_center, width, height]
                x_center, y_center, width, height = box
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
            else:  # Assume already in corner format
                x1, y1, x2, y2 = map(int, box[:4])
            
            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                self.config.box_thickness
            )
            
            # Draw label
            if self.config.show_class_labels or self.config.show_confidence:
                # Prepare label text
                label_parts = []
                
                if self.config.show_tracking and i < len(track_ids):
                    label_parts.append(f"ID:{track_ids[i]}")
                
                if self.config.show_class_labels and class_id < len(self.class_names):
                    label_parts.append(self.class_names[class_id])
                
                if self.config.show_confidence:
                    label_parts.append(f"{score:.2f}")
                
                label = " ".join(label_parts)
                
                # Calculate text size
                if hasattr(self, 'font') and self.font:
                    # Use PIL for text rendering
                    text_bbox = draw.textbbox((0, 0), label, font=self.font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    # Fallback to OpenCV
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.text_scale,
                        self.config.text_thickness
                    )
                
                # Draw label background
                label_y1 = max(y1 - text_height - 5, 0)
                label_y2 = y1
                label_x1 = x1
                label_x2 = x1 + text_width + 10
                
                cv2.rectangle(
                    image,
                    (label_x1, label_y1),
                    (label_x2, label_y2),
                    color,
                    -1  # Filled
                )
                
                # Draw label text
                if hasattr(self, 'font') and self.font:
                    # Convert back to PIL for text
                    temp_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    temp_draw = ImageDraw.Draw(temp_image)
                    temp_draw.text(
                        (x1 + 5, label_y1 + 2),
                        label,
                        font=self.font,
                        fill=(255, 255, 255)
                    )
                    image = cv2.cvtColor(np.array(temp_image), cv2.COLOR_RGB2BGR)
                else:
                    cv2.putText(
                        image,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.text_scale,
                        (255, 255, 255),
                        self.config.text_thickness,
                        cv2.LINE_AA
                    )
        
        return image
    
    def _draw_performance_overlay(self, image: np.ndarray,
                                detections: Dict[str, Any]) -> np.ndarray:
        """Draw performance metrics overlay."""
        if not self.config.show_performance_stats:
            return image
        
        height, width = image.shape[:2]
        overlay = np.zeros((150, width, 3), dtype=np.uint8)
        
        # Get performance metrics
        fps = self._calculate_fps()
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        # Add metadata from detections
        num_detections = detections.get('metadata', {}).get('num_detections', 0)
        processing_time = detections.get('metadata', {}).get('processing_time_ms', 0)
        
        # Create text lines
        lines = [
            f"FPS: {fps:.1f}",
            f"Inference: {avg_inference_time:.1f}ms",
            f"Processing: {processing_time:.1f}ms",
            f"Detections: {num_detections}",
        ]
        
        # Add tracking info if available
        if 'tracking' in detections:
            num_tracks = detections['tracking'].get('num_active_tracks', 0)
            lines.append(f"Active Tracks: {num_tracks}")
        
        # Draw text
        y_offset = 30
        for i, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Create mini performance plot
        if len(self.fps_history) > 1:
            plot_height = 100
            plot_width = 200
            plot_x = width - plot_width - 10
            plot_y = 25
            
            # Create plot background
            cv2.rectangle(
                overlay,
                (plot_x, plot_y),
                (plot_x + plot_width, plot_y + plot_height),
                (50, 50, 50),
                -1
            )
            
            # Normalize FPS values
            fps_values = list(self.fps_history)
            if fps_values:
                min_fps = min(fps_values)
                max_fps = max(fps_values)
                if max_fps > min_fps:
                    normalized = [(f - min_fps) / (max_fps - min_fps) for f in fps_values]
                else:
                    normalized = [0.5] * len(fps_values)
                
                # Draw plot
                for i in range(1, len(normalized)):
                    x1 = plot_x + int((i - 1) * plot_width / len(normalized))
                    y1 = plot_y + plot_height - int(normalized[i - 1] * plot_height)
                    x2 = plot_x + int(i * plot_width / len(normalized))
                    y2 = plot_y + plot_height - int(normalized[i] * plot_height)
                    
                    cv2.line(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        self.config.plot_colors[0],
                        2
                    )
                
                # Add plot label
                cv2.putText(
                    overlay,
                    f"FPS: {min_fps:.0f}-{max_fps:.0f}",
                    (plot_x + 5, plot_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        # Blend overlay with image
        result = np.vstack([overlay, image])
        
        return result
    
    def _draw_debug_info(self, image: np.ndarray,
                        detections: Dict[str, Any]) -> np.ndarray:
        """Draw debug information."""
        if 'debug_info' not in detections:
            return image
        
        debug_info = detections['debug_info']
        height, width = image.shape[:2]
        
        # Create debug overlay
        overlay_height = min(200, height // 3)
        overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
        
        # Add debug information
        y_offset = 20
        for i, (key, value) in enumerate(debug_info.items()):
            if isinstance(value, (int, float)):
                line = f"{key}: {value:.4f}"
            else:
                line = f"{key}: {value}"
            
            cv2.putText(
                overlay,
                line,
                (10, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Blend with image
        result = np.vstack([image, overlay])
        
        return result
    
    def _draw_robot_state(self, image: np.ndarray,
                         detections: Dict[str, Any]) -> np.ndarray:
        """Draw robot state information."""
        if 'robot_state' not in detections:
            return image
        
        robot_state = detections['robot_state']
        height, width = image.shape[:2]
        
        # Create robot state overlay
        overlay_height = 100
        overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
        
        # Draw robot state
        lines = [
            f"Robot State: {robot_state.get('status', 'unknown')}",
            f"Battery: {robot_state.get('battery', 0):.1f}%",
            f"Temperature: {robot_state.get('temperature', 0):.1f}°C",
            f"Uptime: {robot_state.get('uptime', 0):.0f}s",
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (10, 25 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Blend with image
        result = np.vstack([image, overlay])
        
        return result
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        current_time = time.time()
        self.frame_counter += 1
        
        # Calculate FPS
        if hasattr(self, 'last_update_time'):
            time_diff = current_time - self.last_update_time
            
            if time_diff > 1.0:  # Update FPS every second
                fps = self.frame_counter / time_diff
                self.fps_history.append(fps)
                self.frame_counter = 0
                self.last_update_time = current_time
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        if not self.fps_history:
            return 0.0
        return np.mean(list(self.fps_history)[-10:])  # Average of last 10 values
    
    def update_inference_time(self, inference_time_ms: float):
        """Update inference time history."""
        self.inference_times.append(inference_time_ms)
    
    def create_debug_visualization(self, feature_maps: Dict[str, torch.Tensor],
                                 attention_maps: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Create debug visualization of feature maps.
        
        Args:
            feature_maps: Dictionary of feature maps
            attention_maps: Optional attention maps
            
        Returns:
            Debug visualization image
        """
        if not feature_maps:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Create subplot grid
        num_features = len(feature_maps)
        cols = min(4, num_features)
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if num_features > 1 else [axes]
        
        for idx, (name, feature) in enumerate(feature_maps.items()):
            if idx >= len(axes):
                break
            
            # Convert to numpy and normalize
            if isinstance(feature, torch.Tensor):
                feature_np = feature.detach().cpu().numpy()
            else:
                feature_np = feature
            
            # Handle different feature map shapes
            if len(feature_np.shape) == 4:  # [B, C, H, W]
                # Take mean across channels
                feature_np = feature_np[0].mean(axis=0)
            elif len(feature_np.shape) == 3:  # [C, H, W] or [H, W, C]
                if feature_np.shape[0] < feature_np.shape[-1]:  # [C, H, W]
                    feature_np = feature_np.mean(axis=0)
                else:  # [H, W, C]
                    feature_np = feature_np.mean(axis=-1)
            
            # Normalize for visualization
            feature_np = (feature_np - feature_np.min()) / (feature_np.max() - feature_np.min() + 1e-6)
            
            # Plot
            ax = axes[idx]
            im = ax.imshow(feature_np, cmap='viridis')
            ax.set_title(name, fontsize=8)
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(len(feature_maps), len(axes)):
            axes[idx].axis('off')
        
        # Convert to numpy image
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert canvas to numpy array
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_image
    
    def save_visualization(self, image: np.ndarray, output_path: str):
        """Save visualization to file."""
        cv2.imwrite(output_path, image)
        logger.info(f"Visualization saved to {output_path}")

class PerformanceMonitor:
    """Performance monitoring and visualization."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Performance metrics storage
        self.metrics = {
            'fps': deque(maxlen=self.config.performance_window_size),
            'inference_time': deque(maxlen=self.config.performance_window_size),
            'processing_time': deque(maxlen=self.config.performance_window_size),
            'memory_usage': deque(maxlen=self.config.performance_window_size),
            'detection_count': deque(maxlen=self.config.performance_window_size),
        }
        
        # Timers
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.frame_counter = 0
        
        logger.info("PerformanceMonitor initialized")
    
    def update(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        current_time = time.time()
        self.frame_counter += 1
        
        # Update metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Calculate FPS
        time_diff = current_time - self.last_log_time
        if time_diff >= 1.0:  # Log every second
            fps = self.frame_counter / time_diff
            self.metrics['fps'].append(fps)
            
            # Reset counters
            self.frame_counter = 0
            self.last_log_time = current_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p95': np.percentile(values, 95) if len(values) > 1 else values[0],
                    'latest': values[-1] if values else 0,
                }
            else:
                summary[key] = {
                    'mean': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'p95': 0,
                    'latest': 0,
                }
        
        # Add uptime
        summary['uptime'] = time.time() - self.start_time
        summary['total_frames'] = sum(len(v) for v in self.metrics.values())
        
        return summary
    
    def create_performance_plot(self) -> np.ndarray:
        """Create performance plot visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Plot 1: FPS over time
        if self.metrics['fps']:
            axes[0].plot(list(self.metrics['fps']), color=self.config.plot_colors[0])
            axes[0].set_title('FPS Over Time')
            axes[0].set_xlabel('Frame')
            axes[0].set_ylabel('FPS')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Inference time
        if self.metrics['inference_time']:
            axes[1].plot(list(self.metrics['inference_time']), color=self.config.plot_colors[1])
            axes[1].set_title('Inference Time')
            axes[1].set_xlabel('Frame')
            axes[1].set_ylabel('Time (ms)')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Memory usage
        if self.metrics['memory_usage']:
            axes[2].plot(list(self.metrics['memory_usage']), color=self.config.plot_colors[2])
            axes[2].set_title('Memory Usage')
            axes[2].set_xlabel('Frame')
            axes[2].set_ylabel('Memory (MB)')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Detection count
        if self.metrics['detection_count']:
            axes[3].plot(list(self.metrics['detection_count']), color=self.config.plot_colors[3])
            axes[3].set_title('Detection Count')
            axes[3].set_xlabel('Frame')
            axes[3].set_ylabel('Count')
            axes[3].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy image
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot_image
    
    def log_summary(self, interval_seconds: int = 10):
        """Log performance summary at intervals."""
        current_time = time.time()
        
        if current_time - self.last_log_time >= interval_seconds:
            summary = self.get_summary()
            
            logger.info("Performance Summary:")
            logger.info(f"  FPS: {summary['fps']['mean']:.1f} ± {summary['fps']['std']:.1f}")
            logger.info(f"  Inference Time: {summary['inference_time']['mean']:.1f}ms")
            logger.info(f"  Processing Time: {summary['processing_time']['mean']:.1f}ms")
            logger.info(f"  Memory Usage: {summary['memory_usage']['mean']:.1f}MB")
            logger.info(f"  Detection Count: {summary['detection_count']['mean']:.1f}")
            logger.info(f"  Uptime: {summary['uptime']:.0f}s")
            
            self.last_log_time = current_time
    
    def reset(self):
        """Reset performance metrics."""
        for values in self.metrics.values():
            values.clear()
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.frame_counter = 0
        
        logger.info("PerformanceMonitor reset")

class DebugVisualizer:
    """Debug visualization for mHC layers and model internals."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Debug data storage
        self.mhc_activations = []
        self.attention_maps = []
        self.gradient_norms = []
        
        logger.info("DebugVisualizer initialized")
    
    def record_mhc_activation(self, layer_name: str, activation: torch.Tensor):
        """Record MHC layer activation."""
        self.mhc_activations.append({
            'layer': layer_name,
            'activation': activation.detach().cpu(),
            'timestamp': time.time()
        })
        
        # Keep only recent activations
        if len(self.mhc_activations) > 100:
            self.mhc_activations.pop(0)
    
    def record_attention_map(self, attention_map: torch.Tensor):
        """Record attention map."""
        self.attention_maps.append({
            'attention': attention_map.detach().cpu(),
            'timestamp': time.time()
        })
        
        if len(self.attention_maps) > 50:
            self.attention_maps.pop(0)
    
    def create_activation_visualization(self) -> np.ndarray:
        """Create visualization of MHC activations."""
        if not self.mhc_activations:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Get latest activations
        latest_activations = self.mhc_activations[-min(4, len(self.mhc_activations)):]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, activation_data in enumerate(latest_activations):
            if idx >= len(axes):
                break
            
            activation = activation_data['activation']
            layer_name = activation_data['layer']
            
            # Flatten activation for histogram
            if activation.dim() > 1:
                activation_flat = activation.flatten().numpy()
            else:
                activation_flat = activation.numpy()
            
            # Create histogram
            ax = axes[idx]
            ax.hist(activation_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f"{layer_name}\nActivation Distribution", fontsize=10)
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Mean: {np.mean(activation_flat):.3f}\nStd: {np.std(activation_flat):.3f}"
            ax.text(0.95, 0.95, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(latest_activations), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy image
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_image
    
    def create_attention_visualization(self) -> np.ndarray:
        """Create visualization of attention maps."""
        if not self.attention_maps:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Get latest attention map
        attention_data = self.attention_maps[-1]
        attention_map = attention_data['attention']
        
        # Handle different attention map shapes
        if attention_map.dim() == 4:  # [B, H, N, N]
            # Take mean across batch and heads
            attention_map = attention_map.mean(dim=(0, 1))
        elif attention_map.dim() == 3:  # [H, N, N]
            attention_map = attention_map.mean(dim=0)
        
        # Convert to numpy and normalize
        attention_np = attention_map.numpy()
        attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-6)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heatmap
        im1 = ax1.imshow(attention_np, cmap='hot', interpolation='nearest')
        ax1.set_title('Attention Heatmap')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Distribution
        attention_flat = attention_np.flatten()
        ax2.hist(attention_flat, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('Attention Distribution')
        ax2.set_xlabel('Attention Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Mean: {np.mean(attention_flat):.3f}\nStd: {np.std(attention_flat):.3f}"
        ax2.text(0.95, 0.95, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy image
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_image