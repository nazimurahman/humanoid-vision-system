# src/deployment/model_server.py
"""
Model serving optimized for production.
Supports Triton Inference Server and TorchServe.
"""

import os
import sys
import json
import time
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import shutil

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.hybrid_vision import HybridVisionSystem
from src.inference.engine import VisionInferenceEngine
from src.utils.logging import setup_logger

logger = setup_logger("model_server")

class ModelFormat(Enum):
    """Supported model formats."""
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TORCH_MODEL = "torch_model"

class ServingBackend(Enum):
    """Supported serving backends."""
    TORCHSERVE = "torchserve"
    TRITON = "triton"
    FASTAPI = "fastapi"
    CUSTOM = "custom"

@dataclass
class ModelConfig:
    """Model configuration for serving."""
    model_name: str = "hybrid_vision"
    model_version: str = "1.0"
    input_shape: List[int] = None  # [batch, channels, height, width]
    output_shapes: Dict[str, List[int]] = None
    max_batch_size: int = 16
    precision: str = "FP16"  # FP32, FP16, INT8
    device: str = "cuda"  # cuda, cpu
    dynamic_batching: bool = True
    instance_count: int = 1
    optimization: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.input_shape is None:
            self.input_shape = [1, 3, 640, 640]
        if self.output_shapes is None:
            self.output_shapes = {
                "detections": [1, 100, 85],  # [batch, max_detections, 5+80]
                "features": [1, 256, 40, 40]
            }
        if self.optimization is None:
            self.optimization = {
                "graph_optimization": True,
                "memory_pool": True,
                "cudnn_benchmark": True
            }

class ModelExporter:
    """Export models to various formats for serving."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """Initialize exporter."""
        self.model_path = model_path
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model
            model_config = self.config.get("model", {})
            self.model = HybridVisionSystem(
                config=model_config,
                num_classes=model_config.get("num_classes", 80),
                use_vit=model_config.get("use_vit", True),
                use_rag=model_config.get("use_rag", False)
            )
            
            # Load weights
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def export_torchscript(self, output_path: str) -> str:
        """Export model to TorchScript."""
        logger.info("Exporting to TorchScript...")
        
        try:
            # Create example input
            example_input = torch.randn(
                1, 3,
                self.config["input_height"],
                self.config["input_width"]
            ).to(self.device)
            
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(self.model, example_input)
            
            # Save TorchScript model
            traced_model.save(output_path)
            
            logger.info(f"TorchScript model saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export TorchScript: {str(e)}")
            raise
    
    def export_onnx(self, output_path: str, opset_version: int = 13) -> str:
        """Export model to ONNX."""
        logger.info("Exporting to ONNX...")
        
        try:
            # Create example input
            example_input = torch.randn(
                1, 3,
                self.config["input_height"],
                self.config["input_width"]
            ).to(self.device)
            
            # Define input/output names
            input_names = ["input_image"]
            output_names = ["detections", "features"]
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                dynamic_axes={
                    'input_image': {0: 'batch_size'},
                    'detections': {0: 'batch_size'},
                    'features': {0: 'batch_size'}
                },
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
            
            # Verify ONNX model
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"ONNX model saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export ONNX: {str(e)}")
            raise
    
    def optimize_for_inference(self, model_format: ModelFormat) -> str:
        """Optimize model for inference."""
        logger.info(f"Optimizing model for {model_format.value}...")
        
        # Create output directory
        output_dir = f"models/optimized/{model_format.value}"
        os.makedirs(output_dir, exist_ok=True)
        
        if model_format == ModelFormat.TORCHSCRIPT:
            output_path = os.path.join(output_dir, "model.pt")
            return self.export_torchscript(output_path)
        
        elif model_format == ModelFormat.ONNX:
            output_path = os.path.join(output_dir, "model.onnx")
            return self.export_onnx(output_path)
        
        elif model_format == ModelFormat.TENSORRT:
            # First export to ONNX, then convert to TensorRT
            onnx_path = self.export_onnx(
                os.path.join(output_dir, "model.onnx")
            )
            return self._convert_to_tensorrt(onnx_path, output_dir)
        
        else:
            raise ValueError(f"Unsupported format: {model_format}")
    
    def _convert_to_tensorrt(self, onnx_path: str, output_dir: str) -> str:
        """Convert ONNX model to TensorRT."""
        try:
            # This requires TensorRT to be installed
            import tensorrt as trt
            
            logger.info("Converting to TensorRT...")
            
            # TensorRT conversion logic
            # Note: This is a simplified example
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:
                
                # Parse ONNX model
                with open(onnx_path, 'rb') as f:
                    if not parser.parse(f.read()):
                        for error in range(parser.num_errors):
                            logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                        raise RuntimeError("Failed to parse ONNX model")
                
                # Build engine
                config = builder.create_builder_config()
                config.max_workspace_size = 1 << 30  # 1GB
                
                if self.config.get("precision") == "FP16":
                    config.set_flag(trt.BuilderFlag.FP16)
                
                engine = builder.build_serialized_network(network, config)
                
                # Save engine
                engine_path = os.path.join(output_dir, "model.engine")
                with open(engine_path, 'wb') as f:
                    f.write(engine)
                
                logger.info(f"TensorRT engine saved to {engine_path}")
                return engine_path
                
        except ImportError:
            logger.warning("TensorRT not installed, skipping TensorRT export")
            return onnx_path
        except Exception as e:
            logger.error(f"Failed to convert to TensorRT: {str(e)}")
            raise

class TritonModelServer:
    """Triton Inference Server manager."""
    
    def __init__(self, config_path: str = "configs/triton.yaml"):
        """Initialize Triton server manager."""
        self.config = self._load_config(config_path)
        self.process = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Triton configuration."""
        default_config = {
            "triton_path": "/opt/tritonserver/bin/tritonserver",
            "model_repository": "models/triton",
            "http_port": 8000,
            "grpc_port": 8001,
            "metrics_port": 8002,
            "log_verbose": 0,
            "exit_timeout": 30,
            "model_control_mode": "explicit",
            "load_models": "hybrid_vision"
        }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return {**default_config, **config.get("triton", {})}
        except Exception as e:
            logger.warning(f"Failed to load Triton config: {e}, using defaults")
            return default_config
    
    def create_model_repository(self, model_config: ModelConfig):
        """Create Triton model repository structure."""
        repo_path = self.config["model_repository"]
        model_name = model_config.model_name
        model_version = "1"
        
        # Create directory structure
        model_dir = os.path.join(repo_path, model_name, model_version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create config.pbtxt
        config_content = self._generate_triton_config(model_config)
        config_path = os.path.join(repo_path, model_name, "config.pbtxt")
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created Triton model repository at {repo_path}")
        
        # Copy model files
        # This assumes model files are already exported
        model_files = {
            "torchscript": "model.pt",
            "onnx": "model.onnx",
            "tensorrt": "model.engine"
        }
        
        for fmt, filename in model_files.items():
            src = f"models/optimized/{fmt}/{filename}"
            if os.path.exists(src):
                shutil.copy(src, model_dir)
                logger.info(f"Copied {fmt} model to repository")
    
    def _generate_triton_config(self, model_config: ModelConfig) -> str:
        """Generate Triton configuration file."""
        config = f"""
name: "{model_config.model_name}"
platform: "{self._get_platform(model_config)}"
max_batch_size: {model_config.max_batch_size}
"""
        
        if model_config.dynamic_batching:
            config += """
dynamic_batching {{
    preferred_batch_size: [1, 2, 4, 8, 16]
    max_queue_delay_microseconds: 100000
}}
"""
        
        # Input configuration
        config += """
input [
  {{
    name: "input_image"
    data_type: {data_type}
    dims: {dims}
  }}
]
""".format(
    data_type="TYPE_FP16" if model_config.precision == "FP16" else "TYPE_FP32",
    dims=str(model_config.input_shape[1:])  # Remove batch dimension
)
        
        # Output configuration
        config += """
output [
  {{
    name: "detections"
    data_type: TYPE_FP32
    dims: [-1, 100, 85]
  }},
  {{
    name: "features"
    data_type: TYPE_FP32
    dims: [-1, 256, 40, 40]
  }}
]
"""
        
        # Instance groups
        config += """
instance_group [
  {{
    count: {count}
    kind: KIND_GPU
  }}
]
""".format(count=model_config.instance_count)
        
        # Optimization
        if model_config.optimization.get("graph_optimization", False):
            config += """
optimization {{
  graph {{
    level: 1
  }}
}}
"""
        
        return config
    
    def _get_platform(self, model_config: ModelConfig) -> str:
        """Get Triton platform string."""
        # This would depend on the model format
        return "pytorch_libtorch"  # For TorchScript
    
    def start(self):
        """Start Triton server."""
        logger.info("Starting Triton Inference Server...")
        
        try:
            # Build command
            cmd = [
                self.config["triton_path"],
                f"--model-repository={self.config['model_repository']}",
                f"--http-port={self.config['http_port']}",
                f"--grpc-port={self.config['grpc_port']}",
                f"--metrics-port={self.config['metrics_port']}",
                f"--log-verbose={self.config['log_verbose']}",
                f"--exit-timeout={self.config['exit_timeout']}",
                f"--model-control-mode={self.config['model_control_mode']}",
            ]
            
            if self.config.get("load_models"):
                cmd.append(f"--load-model={self.config['load_models']}")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Triton server started with PID {self.process.pid}")
            
            # Wait for server to be ready
            self._wait_for_server_ready()
            
        except Exception as e:
            logger.error(f"Failed to start Triton server: {str(e)}")
            raise
    
    def _wait_for_server_ready(self, timeout: int = 60):
        """Wait for Triton server to be ready."""
        import requests
        
        start_time = time.time()
        health_url = f"http://localhost:{self.config['http_port']}/v2/health/ready"
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("Triton server is ready")
                    return
            except requests.exceptions.ConnectionError:
                pass
            
            time.sleep(1)
        
        raise TimeoutError("Triton server failed to start within timeout")
    
    def stop(self):
        """Stop Triton server."""
        if self.process:
            logger.info("Stopping Triton server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info("Triton server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Triton server did not stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()

class ModelServerManager:
    """Main model server manager."""
    
    def __init__(self, backend: ServingBackend = ServingBackend.TORCHSERVE):
        """Initialize model server manager."""
        self.backend = backend
        self.config = self._load_config()
        self.server = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load server configuration."""
        config_path = f"configs/{self.backend.value}.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}
    
    def deploy_model(self, model_path: str, model_config: ModelConfig):
        """Deploy model to serving backend."""
        logger.info(f"Deploying model to {self.backend.value}...")
        
        # Export model to appropriate format
        exporter = ModelExporter(model_path, self.config)
        exporter.load_model()
        
        if self.backend == ServingBackend.TORCHSERVE:
            # Export to TorchScript for TorchServe
            model_file = exporter.optimize_for_inference(ModelFormat.TORCHSCRIPT)
            self._deploy_torchserve(model_file, model_config)
            
        elif self.backend == ServingBackend.TRITON:
            # Export to appropriate format for Triton
            model_format = ModelFormat.TORCHSCRIPT  # Could be ONNX or TensorRT
            model_file = exporter.optimize_for_inference(model_format)
            
            # Create and start Triton server
            self.server = TritonModelServer()
            self.server.create_model_repository(model_config)
            self.server.start()
            
        elif self.backend == ServingBackend.FASTAPI:
            # Use our custom FastAPI server
            from src.deployment.api_server import VisionAPIServer
            self.server = VisionAPIServer()
            # Server would be started separately
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _deploy_torchserve(self, model_file: str, model_config: ModelConfig):
        """Deploy model to TorchServe."""
        try:
            # Create handler file
            handler_content = """
import torch
import torch.nn.functional as F
import json
import base64
import io
from PIL import Image
import numpy as np

class HybridVisionHandler:
    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        
    def initialize(self, context):
        # Load model
        model_path = context.system_properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = torch.jit.load(model_path + "/model.pt", map_location=self.device)
        self.model.eval()
        self.initialized = True
        
    def preprocess(self, data):
        # Extract image from request
        image_data = data[0].get("body", {}).get("image", "")
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize and normalize
        image = image.resize((640, 640))
        image_np = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        return image_tensor
        
    def inference(self, data):
        with torch.no_grad():
            output = self.model(data)
        return output
        
    def postprocess(self, data):
        # Convert detections to JSON-serializable format
        detections = data[0].cpu().numpy()
        
        # Process detections (simplified)
        results = []
        for i in range(detections.shape[1]):
            if detections[0, i, 4] > 0.5:  # Confidence threshold
                result = {
                    "bbox": detections[0, i, :4].tolist(),
                    "confidence": float(detections[0, i, 4]),
                    "class_id": int(detections[0, i, 5:].argmax())
                }
                results.append(result)
        
        return [json.dumps({"detections": results})]
"""
            
            # Save handler
            handler_path = "models/torchserve/handler.py"
            os.makedirs(os.path.dirname(handler_path), exist_ok=True)
            
            with open(handler_path, 'w') as f:
                f.write(handler_content)
            
            # Create model archive
            from torch_model_archiver.model_packaging import package_model
            
            archive_path = package_model(
                model_name=model_config.model_name,
                model_file=model_file,
                handler=handler_path,
                export_path="models/torchserve",
                version=model_config.model_version
            )
            
            logger.info(f"Created TorchServe archive: {archive_path}")
            
            # Start TorchServe
            self._start_torchserve(archive_path)
            
        except Exception as e:
            logger.error(f"Failed to deploy to TorchServe: {str(e)}")
            raise
    
    def _start_torchserve(self, model_archive: str):
        """Start TorchServe server."""
        try:
            cmd = [
                "torchserve",
                "--start",
                "--model-store", os.path.dirname(model_archive),
                "--models", f"{os.path.basename(model_archive)}",
                "--ts-config", "configs/torchserve_config.properties"
            ]
            
            subprocess.run(cmd, check=True)
            logger.info("TorchServe started")
            
        except Exception as e:
            logger.error(f"Failed to start TorchServe: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        if not self.server:
            return {"status": "not_running"}
        
        # Implementation would depend on backend
        return {
            "backend": self.backend.value,
            "status": "running",
            "models": [self.config.get("model_name", "unknown")]
        }
    
    def stop(self):
        """Stop the model server."""
        if self.server:
            if hasattr(self.server, 'stop'):
                self.server.stop()
            logger.info(f"{self.backend.value} server stopped")

def export_and_serve():
    """Export model and start serving."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Server Manager")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--backend", default="torchserve", 
                       choices=["torchserve", "triton", "fastapi"],
                       help="Serving backend")
    parser.add_argument("--export-only", action="store_true",
                       help="Only export model, don't start server")
    
    args = parser.parse_args()
    
    # Create model configuration
    model_config = ModelConfig(
        model_name="hybrid_vision",
        model_version="1.0",
        input_shape=[1, 3, 640, 640],
        max_batch_size=16,
        precision="FP16" if torch.cuda.is_available() else "FP32",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create server manager
    backend = ServingBackend(args.backend.upper())
    manager = ModelServerManager(backend)
    
    if args.export_only:
        # Just export the model
        exporter = ModelExporter(args.model, manager.config)
        exporter.load_model()
        
        # Export to all formats
        formats = [ModelFormat.TORCHSCRIPT, ModelFormat.ONNX]
        for fmt in formats:
            try:
                output = exporter.optimize_for_inference(fmt)
                logger.info(f"Exported to {fmt.value}: {output}")
            except Exception as e:
                logger.error(f"Failed to export to {fmt.value}: {e}")
    else:
        # Deploy and serve
        manager.deploy_model(args.model, model_config)
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            manager.stop()

if __name__ == "__main__":
    export_and_serve()