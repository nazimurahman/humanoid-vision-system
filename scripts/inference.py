#!/usr/bin/env python3
"""
Real-time inference script for Hybrid Vision System.
Supports images, videos, webcam, and ROS topics.
"""

import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
import time
from queue import Queue
from threading import Thread
import json
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.inference_config import InferenceConfig
from models.hybrid_vision import HybridVisionSystem
from inference.engine import InferenceEngine
from inference.preprocessing import ImagePreprocessor
from inference.postprocessing import DetectionPostprocessor
from inference.visualizer import DetectionVisualizer
from utils.logging import setup_logging, get_logger
from utils.profiler import InferenceProfiler

class VideoStream:
    """Handle video streaming from file or camera."""
    
    def __init__(self, source, buffer_size=64):
        self.source = source
        self.buffer_size = buffer_size
        
        # Open video stream
        if source.isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize buffer
        self.queue = Queue(maxsize=buffer_size)
        self.stopped = False
        
    def start(self):
        """Start video stream thread."""
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self
    
    def update(self):
        """Read frames from video stream."""
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                
                if not ret:
                    self.stopped = True
                    break
                
                self.queue.put(frame)
            else:
                time.sleep(0.01)  # Wait if buffer is full
    
    def read(self):
        """Read next frame from buffer."""
        return self.queue.get()
    
    def more(self):
        """Check if more frames are available."""
        return self.queue.qsize() > 0 or not self.stopped
    
    def stop(self):
        """Stop video stream."""
        self.stopped = True
        self.cap.release()
    
    def release(self):
        """Release video capture."""
        if self.cap.isOpened():
            self.cap.release()

def process_image(image_path, engine, preprocessor, postprocessor, visualizer, config, logger):
    """Process single image."""
    
    logger.info(f"Processing image: {image_path}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Cannot read image: {image_path}")
        return None
    
    # Preprocess
    start_time = time.perf_counter()
    input_tensor = preprocessor.process(image)
    preprocess_time = time.perf_counter() - start_time
    
    # Inference
    start_time = time.perf_counter()
    with torch.no_grad():
        predictions = engine.inference(input_tensor)
    inference_time = time.perf_counter() - start_time
    
    # Postprocess
    start_time = time.perf_counter()
    detections = postprocessor.process(predictions['detections'][0])
    postprocess_time = time.perf_counter() - start_time
    
    # Visualize
    if config.visualize:
        visualized = visualizer.draw_detections(
            image.copy(),
            detections,
            confidence_threshold=config.confidence_threshold
        )
    else:
        visualized = image
    
    # Create result
    result = {
        'image_path': str(image_path),
        'detections': detections,
        'timing': {
            'preprocess_ms': preprocess_time * 1000,
            'inference_ms': inference_time * 1000,
            'postprocess_ms': postprocess_time * 1000,
            'total_ms': (preprocess_time + inference_time + postprocess_time) * 1000
        },
        'num_detections': len(detections),
        'visualized_image': visualized if config.save_output else None
    }
    
    return result

def process_video(video_path, engine, preprocessor, postprocessor, visualizer, config, logger):
    """Process video file."""
    
    logger.info(f"Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    if config.save_output:
        output_path = Path(config.output_dir) / f'output_{Path(video_path).stem}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initialize profiler
    profiler = InferenceProfiler()
    frame_results = []
    frame_count = 0
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = process_image_frame(frame, engine, preprocessor, postprocessor, visualizer, config)
        
        if result:
            frame_results.append(result)
            
            # Write output frame
            if config.save_output:
                out.write(result['visualized_image'])
            
            # Display if enabled
            if config.show:
                cv2.imshow('Detection', result['visualized_image'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Update profiler
            profiler.update(result['timing'])
        
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    if config.save_output:
        out.release()
    if config.show:
        cv2.destroyAllWindows()
    
    # Get profiling results
    profiling_results = profiler.get_results()
    
    return {
        'video_path': str(video_path),
        'total_frames': frame_count,
        'frame_results': frame_results,
        'profiling': profiling_results
    }

def process_webcam(camera_id, engine, preprocessor, postprocessor, visualizer, config, logger):
    """Process webcam stream."""
    
    logger.info(f"Starting webcam stream from camera {camera_id}")
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize profiler
    profiler = InferenceProfiler()
    frame_count = 0
    
    # Warm up
    logger.info("Warming up...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            _ = process_image_frame(frame, engine, preprocessor, postprocessor, visualizer, config)
    
    logger.info("Starting real-time inference. Press 'q' to quit.")
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break
        
        # Process frame
        result = process_image_frame(frame, engine, preprocessor, postprocessor, visualizer, config)
        
        if result:
            # Display
            cv2.imshow('Real-time Detection', result['visualized_image'])
            
            # Update profiler
            profiler.update(result['timing'])
            
            # Print FPS every 30 frames
            if frame_count % 30 == 0:
                fps = 1000.0 / result['timing']['total_ms']
                cv2.setWindowTitle('Real-time Detection', f'Real-time Detection - FPS: {fps:.1f}')
        
        frame_count += 1
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Get profiling results
    profiling_results = profiler.get_results()
    
    logger.info(f"Processed {frame_count} frames")
    logger.info(f"Average FPS: {1000.0 / profiling_results['mean_total_ms']:.1f}")
    
    return profiling_results

def process_image_frame(frame, engine, preprocessor, postprocessor, visualizer, config):
    """Process a single frame."""
    
    # Preprocess
    input_tensor = preprocessor.process(frame)
    
    # Inference
    with torch.no_grad():
        predictions = engine.inference(input_tensor)
    
    # Postprocess
    detections = postprocessor.process(predictions['detections'][0])
    
    # Visualize
    visualized = visualizer.draw_detections(
        frame.copy(),
        detections,
        confidence_threshold=config.confidence_threshold
    )
    
    # Timing (simplified for real-time)
    return {
        'detections': detections,
        'timing': {
            'total_ms': engine.last_inference_time * 1000 if hasattr(engine, 'last_inference_time') else 0
        },
        'visualized_image': visualized
    }

def save_results(results, output_dir, config, logger):
    """Save inference results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detection results
    if 'detections' in results:
        detections_file = output_dir / 'detections.json'
        with open(detections_file, 'w') as f:
            json.dump(results['detections'], f, indent=2)
        logger.info(f"Detections saved to {detections_file}")
    
    # Save timing results
    if 'timing' in results:
        timing_file = output_dir / 'timing.json'
        with open(timing_file, 'w') as f:
            json.dump(results['timing'], f, indent=2)
        logger.info(f"Timing results saved to {timing_file}")
    
    # Save visualized image
    if 'visualized_image' in results and results['visualized_image'] is not None:
        image_file = output_dir / 'output.jpg'
        cv2.imwrite(str(image_file), results['visualized_image'])
        logger.info(f"Visualized image saved to {image_file}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': config.model_path,
        'input_source': config.input,
        'config': {
            'confidence_threshold': config.confidence_threshold,
            'iou_threshold': config.nms_iou_threshold,
            'max_detections': config.max_detections
        }
    }
    
    if 'timing' in results:
        summary['performance'] = results['timing']
    
    if 'num_detections' in results:
        summary['num_detections'] = results['num_detections']
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")

def main(args):
    """Main inference function."""
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    inference_config = InferenceConfig(**config_dict.get('inference', {}))
    
    # Override with command line arguments
    if args.confidence_threshold is not None:
        inference_config.confidence_threshold = args.confidence_threshold
    if args.iou_threshold is not None:
        inference_config.nms_iou_threshold = args.iou_threshold
    if args.max_detections is not None:
        inference_config.max_detections = args.max_detections
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model_config = checkpoint.get('config', {})
    
    model = HybridVisionSystem(
        config=model_config,
        num_classes=args.num_classes or model_config.get('num_classes', 80),
        use_vit=model_config.get('use_vit', True),
        use_rag=model_config.get('use_rag', False)
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(inference_config.device)
    model.eval()
    
    # Create components
    preprocessor = ImagePreprocessor(
        image_size=inference_config.image_size,
        normalize=inference_config.normalize
    )
    
    postprocessor = DetectionPostprocessor(
        confidence_threshold=inference_config.confidence_threshold,
        iou_threshold=inference_config.nms_iou_threshold,
        max_detections=inference_config.max_detections
    )
    
    engine = InferenceEngine(
        model=model,
        config=inference_config,
        device=inference_config.device
    )
    
    visualizer = DetectionVisualizer(
        class_names=inference_config.class_names,
        colors=inference_config.colors
    )
    
    # Process input based on type
    input_path = Path(args.input)
    results = None
    
    if args.input == 'webcam':
        # Webcam inference
        results = process_webcam(
            0, engine, preprocessor, postprocessor, visualizer, args, logger
        )
    
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video inference
        results = process_video(
            input_path, engine, preprocessor, postprocessor, visualizer, args, logger
        )
    
    elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Single image inference
        results = process_image(
            input_path, engine, preprocessor, postprocessor, visualizer, args, logger
        )
    
    elif input_path.is_dir():
        # Directory of images
        image_files = list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.jpeg')) + \
                     list(input_path.glob('*.png'))
        
        all_results = []
        for image_file in image_files:
            result = process_image(
                image_file, engine, preprocessor, postprocessor, visualizer, args, logger
            )
            if result:
                all_results.append(result)
        
        results = {
            'batch_results': all_results,
            'num_images': len(all_results)
        }
    
    else:
        logger.error(f"Unsupported input type: {args.input}")
        return
    
    # Save results
    if args.save_output and results is not None:
        save_results(results, args.output_dir, args, logger)
    
    logger.info("Inference completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with Hybrid Vision System')
    
    # Input arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input source: path to image/video, directory, or "webcam"')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/inference',
                       help='Directory to save output results')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output images/videos')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize detections')
    parser.add_argument('--show', action='store_true',
                       help='Show output in window (for videos/webcam)')
    
    # Detection parameters
    parser.add_argument('--confidence-threshold', type=float, default=None,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=None,
                       help='IOU threshold for NMS')
    parser.add_argument('--max-detections', type=int, default=None,
                       help='Maximum number of detections per image')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=80,
                       help='Number of classes')
    parser.add_argument('--config', type=str, default='configs/inference.yaml',
                       help='Path to configuration file')
    
    # Performance
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--profile', action='store_true',
                       help='Enable detailed profiling')
    
    args = parser.parse_args()
    
    main(args)