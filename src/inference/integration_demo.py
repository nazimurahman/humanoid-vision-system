# src/inference/integration_demo.py
"""
Complete Inference Pipeline Demo

This script demonstrates the complete inference pipeline:
1. Camera input
2. Image preprocessing
3. Model inference
4. Detection postprocessing
5. Visualization
6. Robot communication
"""

import cv2
import numpy as np
import torch
import time
from typing import Dict, Any, Optional
import logging

from .engine import InferenceEngine, InferenceConfig
from .preprocessing import ImagePreprocessor, VideoStreamer, CameraManager
from .postprocessing import DetectionPostprocessor, PostprocessingConfig
from .visualizer import DetectionVisualizer, VisualizationConfig, VisualizationMode
from .robot_interface import RobotCommunication, RobotConfig, DetectionCommand

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteInferencePipeline:
    """
    Complete inference pipeline for robotic vision.
    
    Integrates all components:
    1. Camera input
    2. Preprocessing
    3. Inference
    4. Postprocessing
    5. Visualization
    6. Robot control
    """
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize complete pipeline.
        
        Args:
            model_path: Path to trained model
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self._init_components(model_path)
        
        # Performance tracking
        self.frame_counter = 0
        self.start_time = time.time()
        self.fps_history = []
        
        logger.info("Complete Inference Pipeline initialized")
    
    def _init_components(self, model_path: str):
        """Initialize all pipeline components."""
        # 1. Inference Engine
        inference_config = InferenceConfig(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_half_precision=True,
            batch_size=1,
            confidence_threshold=0.25,
            iou_threshold=0.45
        )
        self.inference_engine = InferenceEngine(inference_config)
        
        # 2. Image Preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # 3. Camera Manager
        self.camera_manager = CameraManager()
        
        # 4. Detection Postprocessor
        postprocessing_config = PostprocessingConfig(
            nms_iou_threshold=0.45,
            nms_score_threshold=0.25,
            enable_tracking=True,
            max_detections_per_frame=50
        )
        self.postprocessor = DetectionPostprocessor(postprocessing_config)
        
        # 5. Visualization
        visualization_config = VisualizationConfig(
            display_width=1280,
            display_height=720,
            show_fps=True,
            show_tracking=True,
            show_class_labels=True
        )
        self.visualizer = DetectionVisualizer(visualization_config)
        
        # 6. Robot Communication (optional)
        self.robot_enabled = self.config.get('enable_robot', False)
        if self.robot_enabled:
            robot_config = RobotConfig(
                robot_ip=self.config.get('robot_ip', '192.168.1.100'),
                robot_port=self.config.get('robot_port', 5000)
            )
            self.robot_comm = RobotCommunication(robot_config)
            self.robot_comm.connect()
    
    def start_camera(self, camera_id: int = 0):
        """Start camera stream."""
        self.camera_manager.add_camera(camera_id)
        self.camera_manager.start_all()
        logger.info(f"Camera {camera_id} started")
    
    def stop_camera(self):
        """Stop camera stream."""
        self.camera_manager.stop_all()
        logger.info("Camera stopped")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Complete processing results
        """
        start_time = time.time()
        
        # 1. Preprocessing
        processed_tensor = self.preprocessor.process(frame)
        
        # 2. Inference
        inference_results = self.inference_engine.infer(processed_tensor)
        
        # 3. Postprocessing
        detections = self.postprocessor.process(
            inference_results['outputs'],
            image_size=frame.shape[:2]
        )
        
        # 4. Update performance metrics
        processing_time = (time.time() - start_time) * 1000
        detections['metadata']['total_processing_time_ms'] = processing_time
        
        # Update visualizer
        self.visualizer.update_inference_time(processing_time)
        
        # 5. Generate robot commands (if enabled)
        if self.robot_enabled and 'detections' in detections:
            self._generate_robot_commands(detections['detections'])
        
        return detections
    
    def _generate_robot_commands(self, detections: Dict[str, Any]):
        """Generate robot commands from detections."""
        boxes = detections.get('boxes', [])
        class_ids = detections.get('class_ids', [])
        scores = detections.get('scores', [])
        
        for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
            if score < 0.5:  # Only high-confidence detections
                continue
            
            # Convert to robot coordinates (simplified)
            # In real implementation, use camera calibration
            x_center, y_center, width, height = box
            
            # Create detection command
            detection_cmd = DetectionCommand(
                detection_id=i,
                class_id=int(class_id),
                confidence=float(score),
                position=(float(x_center), float(y_center), 0.0),
                action=self._get_action_for_class(class_id),
                priority=1 if class_id == 0 else 2  # Higher priority for people
            )
            
            # Send to robot
            self.robot_comm.send_detection_command(detection_cmd)
    
    def _get_action_for_class(self, class_id: int) -> str:
        """Get appropriate action for detected class."""
        # Simple rule-based action selection
        if class_id == 0:  # person
            return "approach"
        elif class_id in [1, 2, 3, 5, 7]:  # vehicles
            return "avoid"
        elif class_id in [56, 57, 62]:  # furniture
            return "avoid"
        else:
            return "ignore"
    
    def run_realtime_demo(self, camera_id: int = 0, max_frames: int = 1000):
        """
        Run real-time inference demo.
        
        Args:
            camera_id: Camera ID to use
            max_frames: Maximum number of frames to process
        """
        # Start camera
        self.start_camera(camera_id)
        
        logger.info("Starting real-time inference demo")
        logger.info("Press 'q' to quit, 's' to save frame")
        
        try:
            while self.frame_counter < max_frames:
                # Get frame from camera
                frame = self.camera_manager.get_camera_frame(camera_id, preprocess=False)
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                results = self.process_frame(frame)
                
                # Visualize results
                visualization = self.visualizer.visualize_detections(
                    frame,
                    results,
                    mode=VisualizationMode.COMBINED
                )
                
                # Display
                cv2.imshow('Robotic Vision System', visualization)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"frame_{timestamp}.jpg", visualization)
                    logger.info(f"Frame saved: frame_{timestamp}.jpg")
                
                # Update frame counter
                self.frame_counter += 1
                
                # Log performance periodically
                if self.frame_counter % 100 == 0:
                    fps = self.frame_counter / (time.time() - self.start_time)
                    logger.info(f"Processed {self.frame_counter} frames, FPS: {fps:.1f}")
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            self.stop_camera()
            
            if self.robot_enabled:
                self.robot_comm.disconnect()
            
            # Log final statistics
            total_time = time.time() - self.start_time
            avg_fps = self.frame_counter / total_time if total_time > 0 else 0
            
            logger.info("Demo completed")
            logger.info(f"Total frames: {self.frame_counter}")
            logger.info(f"Total time: {total_time:.1f}s")
            logger.info(f"Average FPS: {avg_fps:.1f}")
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None):
        """
        Process video file through the pipeline.
        
        Args:
            video_path: Path to input video file
            output_path: Optional output video path
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare video writer if output specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Visualize
            visualization = self.visualizer.visualize_detections(
                frame,
                results,
                mode=VisualizationMode.COMBINED
            )
            
            # Write to output if specified
            if writer:
                writer.write(visualization)
            
            frame_count += 1
            
            # Log progress
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_count} frames, FPS: {fps_processed:.1f}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        logger.info(f"Video processing completed")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        
        if output_path:
            logger.info(f"Output saved to: {output_path}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_time = time.time() - self.start_time
        
        stats = {
            'frames_processed': self.frame_counter,
            'total_time_seconds': total_time,
            'average_fps': self.frame_counter / total_time if total_time > 0 else 0,
            'inference_stats': self.inference_engine.get_performance_stats(),
        }
        
        # Add robot stats if enabled
        if self.robot_enabled:
            stats['robot_stats'] = self.robot_comm.get_performance_stats()
        
        return stats

def main():
    """Main function for demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robotic Vision System Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--video', type=str,
                       help='Path to input video file')
    parser.add_argument('--output', type=str,
                       help='Output video path')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum frames to process')
    parser.add_argument('--enable-robot', action='store_true',
                       help='Enable robot communication')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline_config = {
        'enable_robot': args.enable_robot,
        'robot_ip': '192.168.1.100',  # Configure as needed
        'robot_port': 5000
    }
    
    pipeline = CompleteInferencePipeline(args.model, pipeline_config)
    
    # Run appropriate mode
    if args.video:
        pipeline.process_video_file(args.video, args.output)
    else:
        pipeline.run_realtime_demo(args.camera, args.max_frames)
    
    # Print final statistics
    stats = pipeline.get_performance_stats()
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    print(f"  {subkey}:")
                    for k, v in subvalue.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    
    print("="*50)

if __name__ == "__main__":
    main()