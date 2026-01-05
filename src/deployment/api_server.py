# src/deployment/api_server.py
"""
FastAPI REST API server for vision system.
Supports:
- Real-time object detection
- Batch processing
- Health monitoring
- Metrics collection
"""

import os
import sys
import json
import base64
import io
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_fastapi_instrumentator import Instrumentator

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.inference.engine import VisionInferenceEngine
from src.utils.logging import setup_logger
from src.deployment.health_check import HealthChecker

# Initialize logger
logger = setup_logger("api_server")

# Initialize metrics
REQUEST_COUNT = prometheus_client.Counter(
    'vision_api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = prometheus_client.Histogram(
    'vision_api_request_duration_seconds',
    'API request latency in seconds',
    ['endpoint']
)

INFERENCE_LATENCY = prometheus_client.Histogram(
    'vision_inference_duration_seconds',
    'Inference latency in seconds',
    ['model_type']
)

# Pydantic models for request/response
class DetectionRequest(BaseModel):
    """Request model for object detection."""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_detections: int = Field(100, ge=1, le=1000)
    return_image: bool = Field(True, description="Return image with detections")
    format: str = Field("json", regex="^(json|image|both)$")
    task: str = Field("detection", regex="^(detection|features|classification)$")

class BatchDetectionRequest(BaseModel):
    """Request model for batch processing."""
    images_base64: List[str] = []
    image_urls: List[str] = []
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    batch_size: int = Field(4, ge=1, le=16)
    async_processing: bool = Field(False, description="Process asynchronously")

class DetectionResult(BaseModel):
    """Response model for a single detection."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    attributes: Optional[Dict[str, Any]] = None

class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    success: bool
    detections: List[DetectionResult]
    inference_time_ms: float
    image_size: Optional[List[int]] = None  # [height, width, channels]
    image_base64: Optional[str] = None
    timestamp: str
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_mb: Dict[str, float]
    system_load: Dict[str, float]
    uptime_seconds: float

class MetricsResponse(BaseModel):
    """Response model for metrics."""
    request_counts: Dict[str, int]
    average_latency_ms: Dict[str, float]
    error_rates: Dict[str, float]
    system_metrics: Dict[str, Any]

class VisionAPIServer:
    """Main API server class."""
    
    def __init__(self, config_path: str = "configs/inference.yaml"):
        """Initialize API server."""
        self.app = FastAPI(
            title="Hybrid Vision System API",
            description="Real-time object detection and vision processing for humanoid robots",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.inference_engine = None
        self.health_checker = None
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
        
        # State tracking
        self.start_time = time.time()
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "endpoint_stats": {}
        }
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Startup/shutdown handlers
        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_event_handler("shutdown", self.shutdown_event)
        
        # Instrument for Prometheus
        Instrumentator().instrument(self.app).expose(self.app)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config.get("api_server", {})
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {
                "host": "0.0.0.0",
                "port": 8000,
                "model_path": "models/vision_model.pt",
                "max_workers": 4,
                "max_request_size": 100 * 1024 * 1024,  # 100MB
                "rate_limit": 100,  # requests per minute
                "enable_cors": True,
                "cors_origins": ["*"]
            }
    
    def _setup_middleware(self):
        """Setup middleware for the API."""
        # CORS middleware
        if self.config.get("enable_cors", True):
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.get("cors_origins", ["*"]),
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # GZip middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for request tracking
        @self.app.middleware("http")
        async def track_requests(request, call_next):
            """Middleware to track request metrics."""
            endpoint = request.url.path
            method = request.method
            
            # Track request start time
            start_time = time.time()
            
            try:
                response = await call_next(request)
                
                # Track successful request
                REQUEST_COUNT.labels(
                    endpoint=endpoint,
                    method=method,
                    status=response.status_code
                ).inc()
                
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
                
                # Update stats
                self.request_stats["total_requests"] += 1
                self.request_stats["successful_requests"] += 1
                
                if endpoint not in self.request_stats["endpoint_stats"]:
                    self.request_stats["endpoint_stats"][endpoint] = {
                        "count": 0,
                        "total_time": 0,
                        "errors": 0
                    }
                
                stats = self.request_stats["endpoint_stats"][endpoint]
                stats["count"] += 1
                stats["total_time"] += time.time() - start_time
                
                return response
                
            except Exception as e:
                # Track failed request
                REQUEST_COUNT.labels(
                    endpoint=endpoint,
                    method=method,
                    status=500
                ).inc()
                
                self.request_stats["total_requests"] += 1
                self.request_stats["failed_requests"] += 1
                
                if endpoint in self.request_stats["endpoint_stats"]:
                    self.request_stats["endpoint_stats"][endpoint]["errors"] += 1
                
                logger.error(f"Request failed: {endpoint} - {str(e)}")
                raise
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", tags=["Root"])
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "Hybrid Vision System API",
                "version": "1.0.0",
                "description": "Real-time vision processing for humanoid robots",
                "endpoints": {
                    "/detect": "Single image detection",
                    "/detect/batch": "Batch processing",
                    "/health": "System health check",
                    "/metrics": "API metrics",
                    "/models": "Available models",
                    "/docs": "API documentation"
                }
            }
        
        @self.app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
        async def detect_image(
            request: DetectionRequest,
            image_file: Optional[UploadFile] = File(None)
        ):
            """
            Detect objects in a single image.
            
            Supports:
            - Base64 encoded image
            - Image URL
            - File upload
            """
            start_time = time.time()
            
            try:
                # Get image from request
                image = await self._get_image_from_request(request, image_file)
                
                if image is None:
                    raise HTTPException(status_code=400, detail="No image provided")
                
                # Run inference
                inference_start = time.time()
                
                with INFERENCE_LATENCY.labels(model_type="hybrid_vision").time():
                    results = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.inference_engine.detect,
                        image,
                        request.confidence_threshold,
                        request.iou_threshold,
                        request.max_detections
                    )
                
                inference_time = (time.time() - inference_start) * 1000
                
                # Prepare response
                detections = []
                for det in results["detections"]:
                    detections.append(DetectionResult(
                        bbox=det["bbox"].tolist() if hasattr(det["bbox"], "tolist") else det["bbox"],
                        confidence=float(det["confidence"]),
                        class_id=int(det["class_id"]),
                        class_name=det["class_name"],
                        attributes=det.get("attributes", {})
                    ))
                
                response_data = {
                    "success": True,
                    "detections": detections,
                    "inference_time_ms": inference_time,
                    "image_size": results.get("image_size"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_info": {
                        "name": self.inference_engine.model_name,
                        "version": self.inference_engine.model_version
                    }
                }
                
                # Add image if requested
                if request.return_image and "visualized_image" in results:
                    img_buffer = io.BytesIO()
                    results["visualized_image"].save(img_buffer, format="JPEG")
                    img_buffer.seek(0)
                    response_data["image_base64"] = base64.b64encode(img_buffer.read()).decode()
                
                return DetectionResponse(**response_data)
                
            except Exception as e:
                logger.error(f"Detection failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
        
        @self.app.post("/detect/batch", tags=["Detection"])
        async def detect_batch(
            request: BatchDetectionRequest,
            background_tasks: BackgroundTasks
        ):
            """
            Detect objects in multiple images (batch processing).
            
            Supports async processing for large batches.
            """
            try:
                images = []
                
                # Load images from base64
                for img_base64 in request.images_base64:
                    try:
                        img_data = base64.b64decode(img_base64)
                        image = Image.open(io.BytesIO(img_data))
                        images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 image: {e}")
                
                # TODO: Add URL loading
                
                if not images:
                    raise HTTPException(status_code=400, detail="No valid images provided")
                
                if request.async_processing:
                    # Process asynchronously in background
                    task_id = f"batch_{int(time.time())}_{len(images)}"
                    background_tasks.add_task(
                        self._process_batch_async,
                        task_id,
                        images,
                        request.confidence_threshold,
                        request.batch_size
                    )
                    
                    return {
                        "success": True,
                        "message": "Batch processing started",
                        "task_id": task_id,
                        "image_count": len(images)
                    }
                else:
                    # Process synchronously
                    results = await self._process_batch_sync(
                        images,
                        request.confidence_threshold,
                        request.batch_size
                    )
                    
                    return {
                        "success": True,
                        "results": results,
                        "image_count": len(images),
                        "processing_time": results.get("total_time", 0)
                    }
                    
            except Exception as e:
                logger.error(f"Batch detection failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")
        
        @self.app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health_check():
            """Check system health and model status."""
            health_data = self.health_checker.check_all()
            
            return HealthResponse(
                status="healthy" if health_data["healthy"] else "unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                model_loaded=self.inference_engine is not None,
                gpu_available=torch.cuda.is_available(),
                gpu_memory_mb=health_data.get("gpu_memory", {}),
                system_load=health_data.get("system_load", {}),
                uptime_seconds=time.time() - self.start_time
            )
        
        @self.app.get("/metrics", response_model=MetricsResponse, tags=["System"])
        async def get_metrics():
            """Get API metrics and statistics."""
            # Calculate endpoint statistics
            endpoint_stats = {}
            for endpoint, stats in self.request_stats["endpoint_stats"].items():
                if stats["count"] > 0:
                    endpoint_stats[endpoint] = {
                        "request_count": stats["count"],
                        "average_latency_ms": (stats["total_time"] / stats["count"]) * 1000,
                        "error_rate": stats["errors"] / stats["count"] if stats["count"] > 0 else 0
                    }
            
            # Get system metrics from health checker
            system_metrics = self.health_checker.get_system_metrics()
            
            return MetricsResponse(
                request_counts={
                    "total": self.request_stats["total_requests"],
                    "successful": self.request_stats["successful_requests"],
                    "failed": self.request_stats["failed_requests"]
                },
                average_latency_ms={
                    endpoint: stats["average_latency_ms"]
                    for endpoint, stats in endpoint_stats.items()
                },
                error_rates={
                    endpoint: stats["error_rate"]
                    for endpoint, stats in endpoint_stats.items()
                },
                system_metrics=system_metrics
            )
        
        @self.app.get("/models", tags=["System"])
        async def list_models():
            """List available models and their status."""
            models = []
            
            if self.inference_engine:
                models.append({
                    "name": self.inference_engine.model_name,
                    "version": self.inference_engine.model_version,
                    "status": "loaded",
                    "type": "hybrid_vision",
                    "input_size": self.inference_engine.input_size,
                    "classes": len(self.inference_engine.class_names),
                    "device": str(self.inference_engine.device)
                })
            
            # Check for other available models
            models_dir = "models"
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith(('.pt', '.pth', '.onnx')):
                        models.append({
                            "name": file,
                            "status": "available",
                            "path": os.path.join(models_dir, file)
                        })
            
            return {"models": models}
        
        @self.app.post("/models/switch", tags=["System"])
        async def switch_model(model_name: str = Form(...)):
            """Switch to a different model."""
            try:
                model_path = os.path.join("models", model_name)
                
                if not os.path.exists(model_path):
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Initialize new model
                new_engine = VisionInferenceEngine(model_path)
                
                # Switch engines
                old_engine = self.inference_engine
                self.inference_engine = new_engine
                
                # Cleanup old engine
                if old_engine:
                    del old_engine
                
                logger.info(f"Switched to model: {model_name}")
                
                return {
                    "success": True,
                    "message": f"Switched to model: {model_name}",
                    "model_info": {
                        "name": new_engine.model_name,
                        "version": new_engine.model_version,
                        "input_size": new_engine.input_size
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to switch model: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")
        
        @self.app.get("/stream/{camera_id}", tags=["Streaming"])
        async def video_stream(camera_id: int = 0):
            """
            Stream video from camera with real-time object detection.
            
            Returns MJPEG stream.
            """
            async def generate_frames():
                # Initialize camera
                cap = cv2.VideoCapture(camera_id)
                
                if not cap.isOpened():
                    yield b"Camera not available"
                    return
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert to PIL Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Run detection
                        results = self.inference_engine.detect(
                            pil_image,
                            confidence_threshold=0.3,
                            iou_threshold=0.5
                        )
                        
                        # Draw detections on frame
                        if "visualized_image" in results:
                            vis_frame = np.array(results["visualized_image"])
                            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                        else:
                            vis_frame = frame
                        
                        # Encode as JPEG
                        ret, buffer = cv2.imencode('.jpg', vis_frame)
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        
                        # Small delay to control frame rate
                        await asyncio.sleep(0.033)  # ~30 FPS
                        
                finally:
                    cap.release()
            
            return StreamingResponse(
                generate_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
    
    async def _get_image_from_request(
        self,
        request: DetectionRequest,
        image_file: Optional[UploadFile]
    ) -> Optional[Image.Image]:
        """Extract image from request in various formats."""
        # Priority: File upload > Base64 > URL
        
        if image_file and image_file.filename:
            try:
                contents = await image_file.read()
                image = Image.open(io.BytesIO(contents))
                return image.convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to read uploaded file: {e}")
        
        if request.image_base64:
            try:
                # Remove data URL prefix if present
                if request.image_base64.startswith("data:image"):
                    request.image_base64 = request.image_base64.split(",")[1]
                
                img_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(img_data))
                return image.convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to decode base64 image: {e}")
        
        if request.image_url:
            # TODO: Implement URL fetching with timeout
            logger.warning("URL fetching not implemented yet")
            
        return None
    
    async def _process_batch_sync(
        self,
        images: List[Image.Image],
        confidence_threshold: float,
        batch_size: int
    ) -> Dict[str, Any]:
        """Process batch of images synchronously."""
        start_time = time.time()
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Run inference on batch
            batch_results = self.inference_engine.detect_batch(
                batch,
                confidence_threshold=confidence_threshold
            )
            
            results.extend(batch_results)
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "total_images": len(images),
            "total_time": total_time,
            "images_per_second": len(images) / total_time if total_time > 0 else 0
        }
    
    async def _process_batch_async(
        self,
        task_id: str,
        images: List[Image.Image],
        confidence_threshold: float,
        batch_size: int
    ):
        """Process batch of images asynchronously in background."""
        logger.info(f"Starting async batch processing: {task_id} ({len(images)} images)")
        
        try:
            results = await self._process_batch_sync(
                images,
                confidence_threshold,
                batch_size
            )
            
            # Save results to file or database
            output_file = f"batch_results/{task_id}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Async batch processing completed: {task_id}")
            
        except Exception as e:
            logger.error(f"Async batch processing failed {task_id}: {str(e)}")
    
    async def startup_event(self):
        """Startup event handler."""
        logger.info("Starting Hybrid Vision API Server...")
        
        try:
            # Initialize inference engine
            model_path = self.config.get("model_path", "models/vision_model.pt")
            self.inference_engine = VisionInferenceEngine(model_path)
            logger.info(f"Inference engine loaded: {model_path}")
            
            # Initialize health checker
            self.health_checker = HealthChecker(self.inference_engine)
            logger.info("Health checker initialized")
            
            # Warm up model
            logger.info("Warming up model...")
            dummy_input = Image.new('RGB', (640, 480), color='white')
            self.inference_engine.detect(dummy_input)
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize API server: {str(e)}")
            raise
    
    async def shutdown_event(self):
        """Shutdown event handler."""
        logger.info("Shutting down API Server...")
        
        # Cleanup resources
        if self.inference_engine:
            del self.inference_engine
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("API Server shutdown complete")

def run_server():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Vision System API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", default="configs/inference.yaml", help="Config file path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    
    args = parser.parse_args()
    
    # Create and run server
    server = VisionAPIServer(args.config)
    
    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    run_server()