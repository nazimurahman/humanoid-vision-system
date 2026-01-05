# src/deployment/grpc_server.py
"""
gRPC server optimized for robot communication.
Low-latency, bidirectional streaming for real-time vision.
"""

import os
import sys
import time
import json
import base64
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from concurrent import futures
import io

import grpc
from grpc_reflection.v1alpha import reflection
import numpy as np
from PIL import Image
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.inference.engine import VisionInferenceEngine
from src.utils.logging import setup_logger

# Import generated protobuf code
try:
    # In production, these would be generated from .proto files
    # For this example, we'll create stub classes
    pass
except ImportError:
    # Create mock protobuf classes for demonstration
    class Detection:
        def __init__(self):
            self.bbox = []
            self.confidence = 0.0
            self.class_id = 0
            self.class_name = ""
            
    class DetectionResponse:
        def __init__(self):
            self.success = False
            self.detections = []
            self.inference_time_ms = 0.0
            self.timestamp = ""
            
    class ImageRequest:
        def __init__(self):
            self.image_data = b""
            self.image_format = "jpeg"
            self.confidence_threshold = 0.5
            self.iou_threshold = 0.5
            
    class StreamRequest:
        def __init__(self):
            self.stream_id = ""
            self.config = {}
            
    class StreamResponse:
        def __init__(self):
            self.frame_id = 0
            self.detections = []
            self.timestamp = ""
            self.frame_data = b""
            
    class RobotCommand:
        def __init__(self):
            self.command = ""
            self.parameters = {}
            
    class CommandResponse:
        def __init__(self):
            self.success = False
            self.message = ""
            self.timestamp = ""
            
    # gRPC service stubs
    class VisionServiceServicer:
        pass

# Initialize logger
logger = setup_logger("grpc_server")

class RobotVisionService(VisionServiceServicer):
    """gRPC service implementation for robot vision."""
    
    def __init__(self, model_path: str = "models/vision_model.pt"):
        """Initialize the gRPC service."""
        # Load inference engine
        self.inference_engine = VisionInferenceEngine(model_path)
        logger.info(f"Loaded inference engine from {model_path}")
        
        # State tracking
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.command_handlers = {
            "ping": self._handle_ping,
            "get_status": self._handle_get_status,
            "switch_model": self._handle_switch_model,
            "update_config": self._handle_update_config,
            "stop_stream": self._handle_stop_stream
        }
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "streams_started": 0,
            "streams_stopped": 0,
            "average_latency_ms": 0.0
        }
        
        # Warm up model
        self._warm_up_model()
        
    def _warm_up_model(self):
        """Warm up the model for faster inference."""
        logger.info("Warming up model...")
        
        # Create dummy input
        dummy_image = Image.new('RGB', (640, 480), color='white')
        
        # Run inference multiple times
        for i in range(3):
            start_time = time.time()
            self.inference_engine.detect(dummy_image)
            latency = (time.time() - start_time) * 1000
            logger.info(f"Warmup {i+1}/3: {latency:.1f}ms")
        
        logger.info("Model warmup completed")
    
    def DetectSingle(self, request, context):
        """Handle single image detection request."""
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # Decode image
            image_data = request.image_data
            if request.image_format == "jpeg":
                image = Image.open(io.BytesIO(image_data))
            else:
                # Handle other formats if needed
                image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Run inference
            results = self.inference_engine.detect(
                image,
                confidence_threshold=request.confidence_threshold,
                iou_threshold=request.iou_threshold
            )
            
            # Build response
            response = DetectionResponse()
            response.success = True
            response.inference_time_ms = (time.time() - start_time) * 1000
            response.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add detections
            for det in results["detections"]:
                detection = Detection()
                detection.bbox.extend(det["bbox"])
                detection.confidence = det["confidence"]
                detection.class_id = det["class_id"]
                detection.class_name = det["class_name"]
                response.detections.append(detection)
            
            # Update stats
            self.stats["average_latency_ms"] = (
                (self.stats["average_latency_ms"] * (self.stats["total_requests"] - 1) +
                 response.inference_time_ms) / self.stats["total_requests"]
            )
            
            logger.debug(f"Single detection: {len(response.detections)} objects in {response.inference_time_ms:.1f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Detection failed: {str(e)}")
            
            response = DetectionResponse()
            response.success = False
            response.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            return response
    
    def DetectBatch(self, request_iterator, context):
        """Handle batch detection with streaming."""
        self.stats["total_requests"] += 1
        
        try:
            images = []
            
            # Collect all images from stream
            for request in request_iterator:
                image_data = request.image_data
                if request.image_format == "jpeg":
                    image = Image.open(io.BytesIO(image_data))
                else:
                    image = Image.open(io.BytesIO(image_data))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                images.append(image)
            
            if not images:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No images provided")
                return
            
            # Process batch
            start_time = time.time()
            batch_results = self.inference_engine.detect_batch(
                images,
                confidence_threshold=0.5  # Default, could be configurable
            )
            
            total_time = (time.time() - start_time) * 1000
            
            # Stream responses back
            for i, results in enumerate(batch_results):
                response = DetectionResponse()
                response.success = True
                response.inference_time_ms = total_time / len(images)
                response.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                for det in results["detections"]:
                    detection = Detection()
                    detection.bbox.extend(det["bbox"])
                    detection.confidence = det["confidence"]
                    detection.class_id = det["class_id"]
                    detection.class_name = det["class_name"]
                    response.detections.append(detection)
                
                yield response
            
            logger.info(f"Batch detection: {len(images)} images in {total_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Batch detection failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Batch detection failed: {str(e)}")
    
    async def StreamDetections(self, request_iterator, context):
        """Bidirectional streaming for real-time detection."""
        self.stats["streams_started"] += 1
        stream_id = f"stream_{self.stats['streams_started']}"
        
        logger.info(f"Starting detection stream: {stream_id}")
        
        try:
            async for request in request_iterator:
                start_time = time.time()
                
                # Decode image
                image_data = request.frame_data
                image = Image.open(io.BytesIO(image_data))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Run inference
                results = self.inference_engine.detect(
                    image,
                    confidence_threshold=0.3,  # Lower for streaming
                    iou_threshold=0.5
                )
                
                # Build response
                response = StreamResponse()
                response.frame_id = request.frame_id
                response.timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")
                
                # Add detections
                for det in results["detections"]:
                    detection = Detection()
                    detection.bbox.extend(det["bbox"])
                    detection.confidence = det["confidence"]
                    detection.class_id = det["class_id"]
                    detection.class_name = det["class_name"]
                    response.detections.append(detection)
                
                # Optionally add visualized frame
                if "visualized_image" in results:
                    img_buffer = io.BytesIO()
                    results["visualized_image"].save(img_buffer, format="JPEG")
                    response.frame_data = img_buffer.getvalue()
                
                inference_time = (time.time() - start_time) * 1000
                
                # Log performance occasionally
                if request.frame_id % 100 == 0:
                    logger.debug(f"Stream {stream_id}: Frame {request.frame_id} in {inference_time:.1f}ms")
                
                yield response
                
        except Exception as e:
            logger.error(f"Stream {stream_id} failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Stream failed: {str(e)}")
        
        finally:
            self.stats["streams_stopped"] += 1
            logger.info(f"Stopped detection stream: {stream_id}")
    
    def HandleCommand(self, request, context):
        """Handle robot commands."""
        self.stats["total_requests"] += 1
        
        try:
            command = request.command
            parameters = json.loads(request.parameters) if request.parameters else {}
            
            logger.info(f"Received command: {command} with params: {parameters}")
            
            # Find command handler
            handler = self.command_handlers.get(command)
            if not handler:
                return CommandResponse(
                    success=False,
                    message=f"Unknown command: {command}",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            # Execute command
            success, message = handler(parameters)
            
            response = CommandResponse()
            response.success = success
            response.message = message
            response.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return response
            
        except Exception as e:
            logger.error(f"Command handling failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Command handling failed: {str(e)}")
            
            response = CommandResponse()
            response.success = False
            response.message = f"Command failed: {str(e)}"
            response.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            return response
    
    def _handle_ping(self, parameters):
        """Handle ping command."""
        return True, "pong"
    
    def _handle_get_status(self, parameters):
        """Handle status request."""
        status = {
            "model_loaded": True,
            "model_name": self.inference_engine.model_name,
            "device": str(self.inference_engine.device),
            "active_streams": len(self.active_streams),
            "total_requests": self.stats["total_requests"],
            "average_latency_ms": self.stats["average_latency_ms"]
        }
        
        return True, json.dumps(status)
    
    def _handle_switch_model(self, parameters):
        """Handle model switching."""
        model_name = parameters.get("model_name")
        if not model_name:
            return False, "No model name provided"
        
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            return False, f"Model not found: {model_path}"
        
        try:
            # Load new model
            new_engine = VisionInferenceEngine(model_path)
            
            # Replace old engine
            old_engine = self.inference_engine
            self.inference_engine = new_engine
            
            # Cleanup
            del old_engine
            
            logger.info(f"Switched to model: {model_name}")
            return True, f"Switched to model: {model_name}"
            
        except Exception as e:
            return False, f"Failed to switch model: {str(e)}"
    
    def _handle_update_config(self, parameters):
        """Handle configuration updates."""
        try:
            # Update inference engine configuration
            if "confidence_threshold" in parameters:
                self.inference_engine.confidence_threshold = parameters["confidence_threshold"]
            
            if "iou_threshold" in parameters:
                self.inference_engine.iou_threshold = parameters["iou_threshold"]
            
            logger.info(f"Updated config: {parameters}")
            return True, "Configuration updated"
            
        except Exception as e:
            return False, f"Failed to update config: {str(e)}"
    
    def _handle_stop_stream(self, parameters):
        """Handle stream stopping."""
        stream_id = parameters.get("stream_id")
        
        if stream_id in self.active_streams:
            self.active_streams[stream_id].cancel()
            del self.active_streams[stream_id]
            return True, f"Stopped stream: {stream_id}"
        else:
            return False, f"Stream not found: {stream_id}"
    
    def GetStats(self, request, context):
        """Get server statistics."""
        stats_response = {
            "total_requests": self.stats["total_requests"],
            "streams_started": self.stats["streams_started"],
            "streams_stopped": self.stats["streams_stopped"],
            "average_latency_ms": self.stats["average_latency_ms"],
            "active_streams": len(self.active_streams),
            "model_info": {
                "name": self.inference_engine.model_name,
                "device": str(self.inference_engine.device)
            }
        }
        
        return json.dumps(stats_response)

class RobotGRPCServer:
    """Main gRPC server class."""
    
    def __init__(self, config_path: str = "configs/grpc_server.yaml"):
        """Initialize gRPC server."""
        self.config = self._load_config(config_path)
        self.server = None
        self.service = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration."""
        import yaml
        
        default_config = {
            "host": "0.0.0.0",
            "port": 50051,
            "max_workers": 10,
            "max_message_length": 100 * 1024 * 1024,  # 100MB
            "model_path": "models/vision_model.pt",
            "ssl_enabled": False,
            "ssl_cert": None,
            "ssl_key": None,
            "enable_reflection": True,
            "enable_health_check": True
        }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return {**default_config, **config.get("grpc_server", {})}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return default_config
    
    def start(self):
        """Start the gRPC server."""
        logger.info("Starting Robot gRPC Server...")
        
        try:
            # Create server
            server_options = [
                ('grpc.max_send_message_length', self.config["max_message_length"]),
                ('grpc.max_receive_message_length', self.config["max_message_length"]),
                ('grpc.max_metadata_size', 16 * 1024),  # 16KB
                ('grpc.enable_retries', 1),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
            ]
            
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.config["max_workers"]),
                options=server_options
            )
            
            # Create and add service
            self.service = RobotVisionService(self.config["model_path"])
            
            # Add service to server (using mock for demonstration)
            # In real implementation, you would use add_VisionServiceServicer_to_server
            
            # Enable reflection
            if self.config["enable_reflection"]:
                service_names = []
                reflection.enable_server_reflection(service_names, self.server)
            
            # Add health checking service
            if self.config["enable_health_check"]:
                # Add health service
                pass
            
            # Bind server
            address = f"{self.config['host']}:{self.config['port']}"
            self.server.add_insecure_port(address)
            
            # Start server
            self.server.start()
            logger.info(f"gRPC Server started on {address}")
            
            # Keep server running
            self.server.wait_for_termination()
            
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {str(e)}")
            raise
    
    def stop(self):
        """Stop the gRPC server."""
        logger.info("Stopping gRPC Server...")
        
        if self.server:
            self.server.stop(grace=5)  # 5 second grace period
        
        logger.info("gRPC Server stopped")

def run_server():
    """Run the gRPC server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot Vision gRPC Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50051, help="Port to bind to")
    parser.add_argument("--config", default="configs/grpc_server.yaml", help="Config file")
    
    args = parser.parse_args()
    
    server = RobotGRPCServer(args.config)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        server.stop()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        server.stop()
        raise

if __name__ == "__main__":
    run_server()