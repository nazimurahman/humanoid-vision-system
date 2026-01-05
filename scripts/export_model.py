#!/usr/bin/env python3
"""
Model export script for Hybrid Vision System.
Exports to various formats: TorchScript, ONNX, TensorRT, CoreML.
"""

import os
import sys
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.model_config import ModelConfig
from models.hybrid_vision import HybridVisionSystem
from utils.logging import setup_logging, get_logger

class ModelExporter:
    """Export model to different formats."""
    
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_torchscript(self, example_input, optimize=True):
        """Export to TorchScript."""
        
        self.logger.info("Exporting to TorchScript...")
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    self.model,
                    example_input,
                    strict=False
                )
            
            # Optimize if requested
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save the model
            output_path = self.output_dir / 'model.torchscript.pt'
            traced_model.save(str(output_path))
            
            # Test the exported model
            with torch.no_grad():
                original_output = self.model(example_input)
                traced_output = traced_model(example_input)
            
            # Check output consistency
            self._check_output_consistency(
                original_output['detections'],
                traced_output['detections'],
                'TorchScript'
            )
            
            self.logger.info(f"TorchScript model saved to {output_path}")
            
            return {
                'format': 'torchscript',
                'path': str(output_path),
                'input_shape': list(example_input.shape),
                'optimized': optimize
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export to TorchScript: {e}")
            raise
    
    def export_onnx(self, example_input, opset_version=13, dynamic_axes=None):
        """Export to ONNX format."""
        
        self.logger.info("Exporting to ONNX...")
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Define output path
            output_path = self.output_dir / 'model.onnx'
            
            # Default dynamic axes for batch dimension
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'detections': {0: 'batch_size'}
                }
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['detections', 'features'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Verify the ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Test with ONNX Runtime
            self._test_onnx_model(output_path, example_input)
            
            self.logger.info(f"ONNX model saved to {output_path}")
            
            return {
                'format': 'onnx',
                'path': str(output_path),
                'opset_version': opset_version,
                'dynamic_axes': dynamic_axes
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export to ONNX: {e}")
            raise
    
    def export_tensorrt(self, example_input, precision='fp16', max_workspace_size=1 << 30):
        """Export to TensorRT (requires TensorRT installation)."""
        
        self.logger.info(f"Exporting to TensorRT with {precision} precision...")
        
        try:
            import tensorrt as trt
            
            # First export to ONNX
            onnx_path = self.output_dir / 'model.onnx'
            self.export_onnx(example_input)
            
            # TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            
            # Create parser
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Build configuration
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            # Set precision
            if precision == 'fp16':
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                else:
                    self.logger.warning("FP16 not supported on this platform, using FP32")
            
            elif precision == 'int8':
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    # TODO: Add calibration dataset for INT8 quantization
                    self.logger.warning("INT8 calibration not implemented")
                else:
                    self.logger.warning("INT8 not supported on this platform, using FP32")
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            output_path = self.output_dir / f'model.{precision}.trt'
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"TensorRT engine saved to {output_path}")
            
            return {
                'format': 'tensorrt',
                'path': str(output_path),
                'precision': precision,
                'max_workspace_size': max_workspace_size
            }
            
        except ImportError:
            self.logger.error("TensorRT not installed. Install with: pip install tensorrt")
            raise
        except Exception as e:
            self.logger.error(f"Failed to export to TensorRT: {e}")
            raise
    
    def export_coreml(self, example_input):
        """Export to CoreML format (for iOS/macOS)."""
        
        self.logger.info("Exporting to CoreML...")
        
        try:
            import coremltools as ct
            
            # First export to ONNX
            onnx_path = self.output_dir / 'model.onnx'
            self.export_onnx(example_input)
            
            # Convert ONNX to CoreML
            mlmodel = ct.converters.onnx.convert(
                model=str(onnx_path),
                minimum_ios_deployment_target='13'
            )
            
            # Add metadata
            mlmodel.author = 'Humanoid Vision System'
            mlmodel.license = 'MIT'
            mlmodel.short_description = 'Hybrid Vision System for robotic perception'
            mlmodel.input_description['input'] = 'Input image'
            mlmodel.output_description['detections'] = 'Object detections'
            
            # Save CoreML model
            output_path = self.output_dir / 'model.mlmodel'
            mlmodel.save(str(output_path))
            
            self.logger.info(f"CoreML model saved to {output_path}")
            
            return {
                'format': 'coreml',
                'path': str(output_path),
                'minimum_ios_version': '13'
            }
            
        except ImportError:
            self.logger.error("CoreML Tools not installed. Install with: pip install coremltools")
            raise
        except Exception as e:
            self.logger.error(f"Failed to export to CoreML: {e}")
            raise
    
    def export_openvino(self, example_input, precision='FP16'):
        """Export to OpenVINO IR format."""
        
        self.logger.info(f"Exporting to OpenVINO with {precision} precision...")
        
        try:
            from openvino.tools import mo
            from openvino.runtime import Core
            
            # First export to ONNX
            onnx_path = self.output_dir / 'model.onnx'
            self.export_onnx(example_input)
            
            # Convert to OpenVINO IR
            ov_model = mo.convert_model(
                str(onnx_path),
                compress_to_fp16=(precision == 'FP16')
            )
            
            # Save OpenVINO model
            output_path = self.output_dir / 'model.xml'
            ov_model.serialize(str(output_path), str(output_path.with_suffix('.bin')))
            
            # Test the model
            ie = Core()
            compiled_model = ie.compile_model(model=ov_model, device_name='CPU')
            
            # Run inference
            input_tensor = example_input.cpu().numpy()
            result = compiled_model(input_tensor)
            
            self.logger.info(f"OpenVINO model saved to {output_path}")
            
            return {
                'format': 'openvino',
                'path': str(output_path),
                'precision': precision
            }
            
        except ImportError:
            self.logger.error("OpenVINO not installed. Install with: pip install openvino")
            raise
        except Exception as e:
            self.logger.error(f"Failed to export to OpenVINO: {e}")
            raise
    
    def export_all(self, example_input, formats=None):
        """Export to all requested formats."""
        
        if formats is None:
            formats = ['torchscript', 'onnx', 'tensorrt']
        
        results = {}
        
        for fmt in formats:
            try:
                if fmt == 'torchscript':
                    results['torchscript'] = self.export_torchscript(example_input)
                
                elif fmt == 'onnx':
                    results['onnx'] = self.export_onnx(example_input)
                
                elif fmt == 'tensorrt':
                    results['tensorrt'] = self.export_tensorrt(example_input)
                
                elif fmt == 'coreml':
                    results['coreml'] = self.export_coreml(example_input)
                
                elif fmt == 'openvino':
                    results['openvino'] = self.export_openvino(example_input)
                
                else:
                    self.logger.warning(f"Unsupported format: {fmt}")
                    
            except Exception as e:
                self.logger.error(f"Failed to export to {fmt}: {e}")
                continue
        
        return results
    
    def _check_output_consistency(self, original, exported, format_name, rtol=1e-3, atol=1e-5):
        """Check consistency between original and exported model outputs."""
        
        if isinstance(original, dict) and isinstance(exported, dict):
            for key in original:
                if key in exported:
                    self._check_output_consistency(
                        original[key], exported[key], f"{format_name}.{key}", rtol, atol
                    )
        elif torch.is_tensor(original) and torch.is_tensor(exported):
            if not torch.allclose(original, exported, rtol=rtol, atol=atol):
                max_diff = torch.max(torch.abs(original - exported)).item()
                self.logger.warning(
                    f"Output mismatch in {format_name}: max_diff={max_diff:.6f}"
                )
            else:
                self.logger.info(f"Output consistency check passed for {format_name}")
        else:
            self.logger.warning(f"Cannot compare outputs for {format_name}")
    
    def _test_onnx_model(self, onnx_path, example_input):
        """Test ONNX model with ONNX Runtime."""
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path))
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_data = example_input.cpu().numpy()
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        # Get original outputs for comparison
        with torch.no_grad():
            original_outputs = self.model(example_input)
        
        self.logger.info(f"ONNX Runtime inference successful. Output shapes: {[o.shape for o in outputs]}")
        
        return outputs
    
    def save_export_info(self, export_results):
        """Save export information to JSON file."""
        
        export_info = {
            'timestamp': str(torch.datetime.now()),
            'model_name': self.config.model_name,
            'input_shape': list(self.config.example_input_shape),
            'formats_exported': list(export_results.keys()),
            'export_results': export_results,
            'config': {
                'use_vit': self.config.use_vit,
                'use_rag': self.config.use_rag,
                'num_classes': self.config.num_classes
            }
        }
        
        info_path = self.output_dir / 'export_info.json'
        with open(info_path, 'w') as f:
            json.dump(export_info, f, indent=2)
        
        self.logger.info(f"Export information saved to {info_path}")

def load_model(checkpoint_path, config_path=None, device='cuda'):
    """Load model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config = ModelConfig(**config_dict.get('model', {}))
    elif 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        raise ValueError("Model configuration not found in checkpoint or config file")
    
    # Create model
    model = HybridVisionSystem(
        config=model_config,
        num_classes=model_config.num_classes,
        use_vit=model_config.use_vit,
        use_rag=model_config.use_rag
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, model_config

def create_example_input(batch_size=1, image_size=(416, 416), device='cuda'):
    """Create example input tensor for export."""
    
    return torch.randn(batch_size, 3, image_size[0], image_size[1]).to(device)

def main(args):
    """Main export function."""
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(f"Starting model export from {args.checkpoint}")
    
    # Load model
    model, model_config = load_model(
        args.checkpoint,
        args.config,
        args.device
    )
    
    # Create example input
    example_input = create_example_input(
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device
    )
    
    # Create exporter configuration
    exporter_config = {
        'output_dir': args.output_dir,
        'model_name': args.model_name or Path(args.checkpoint).stem,
        'example_input_shape': list(example_input.shape),
        'use_vit': model_config.use_vit,
        'use_rag': model_config.use_rag,
        'num_classes': model_config.num_classes
    }
    
    # Create exporter
    exporter = ModelExporter(model, exporter_config, logger)
    
    # Export to requested formats
    formats = args.formats.split(',') if args.formats else ['torchscript', 'onnx']
    
    logger.info(f"Exporting to formats: {formats}")
    
    try:
        export_results = exporter.export_all(example_input, formats)
        
        # Save export information
        exporter.save_export_info(export_results)
        
        logger.info("Model export completed successfully")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EXPORT SUMMARY")
        logger.info("="*50)
        for fmt, info in export_results.items():
            logger.info(f"{fmt.upper()}: {info['path']}")
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Hybrid Vision System model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Export formats
    parser.add_argument('--formats', type=str, default='torchscript,onnx',
                       help='Comma-separated list of formats to export: torchscript,onnx,tensorrt,coreml,openvino')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='models/exported',
                       help='Directory to save exported models')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Name for exported model')
    
    # Input configuration
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for example input')
    parser.add_argument('--image-size', type=int, nargs=2, default=[416, 416],
                       help='Input image size (height width)')
    
    # Model configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run export on (cuda or cpu)')
    
    # Format-specific options
    parser.add_argument('--opset-version', type=int, default=13,
                       help='ONNX opset version')
    parser.add_argument('--tensorrt-precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='TensorRT precision')
    parser.add_argument('--openvino-precision', type=str, default='FP16',
                       choices=['FP32', 'FP16'],
                       help='OpenVINO precision')
    
    # Optimization
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize exported models for inference')
    parser.add_argument('--dynamic-batch', action='store_true',
                       help='Enable dynamic batch size')
    
    args = parser.parse_args()
    
    main(args)