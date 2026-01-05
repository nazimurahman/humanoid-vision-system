#!/usr/bin/env python3
"""
Evaluation script for Hybrid Vision System.
Computes comprehensive metrics including mAP, inference speed, and stability metrics.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from models.hybrid_vision import HybridVisionSystem
from inference.engine import InferenceEngine
from inference.postprocessing import DetectionPostprocessor
from data.dataset import COCOVisionDataset
from data.transforms import create_val_transforms
from utils.metrics import DetectionMetrics, StabilityAnalyzer
from utils.logging import setup_logging, get_logger

def evaluate_model(model, dataset, config, logger):
    """Evaluate model on dataset."""
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        config=config,
        device=config.device
    )
    
    # Create postprocessor
    postprocessor = DetectionPostprocessor(
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.nms_iou_threshold,
        max_detections=config.max_detections
    )
    
    # Results storage
    all_predictions = []
    all_targets = []
    inference_times = []
    stability_metrics = []
    
    logger.info(f"Starting evaluation on {len(dataset)} images")
    
    # Evaluation loop
    for idx in tqdm(range(len(dataset)), desc='Evaluating'):
        try:
            # Load image and target
            image, target = dataset[idx]
            
            # Time inference
            start_time = time.perf_counter()
            
            # Run inference
            with torch.no_grad():
                predictions = engine.inference(image.unsqueeze(0))
            
            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)
            
            # Postprocess predictions
            detections = postprocessor.process(predictions['detections'][0])
            
            # Convert to COCO format
            image_id = target['image_id'].item()
            img_info = dataset.get_image_info(image_id)
            
            # Store predictions
            for det in detections:
                coco_pred = {
                    'image_id': int(image_id),
                    'category_id': int(det['class_id']),
                    'bbox': [
                        float(det['bbox'][0]),  # x
                        float(det['bbox'][1]),  # y
                        float(det['bbox'][2] - det['bbox'][0]),  # width
                        float(det['bbox'][3] - det['bbox'][1])   # height
                    ],
                    'score': float(det['confidence']),
                    'area': float((det['bbox'][2] - det['bbox'][0]) * 
                                 (det['bbox'][3] - det['bbox'][1]))
                }
                all_predictions.append(coco_pred)
            
            # Store targets
            for ann in dataset.get_annotations(image_id):
                coco_target = {
                    'image_id': int(image_id),
                    'category_id': int(ann['category_id']),
                    'bbox': [float(x) for x in ann['bbox']],
                    'area': float(ann['area']),
                    'iscrowd': int(ann['iscrowd'])
                }
                all_targets.append(coco_target)
            
            # Collect stability metrics
            if 'stability_metrics' in predictions:
                stability_metrics.append(predictions['stability_metrics'])
                
        except Exception as e:
            logger.warning(f"Error processing image {idx}: {e}")
            continue
    
    # Convert to DataFrames
    predictions_df = pd.DataFrame(all_predictions)
    targets_df = pd.DataFrame(all_targets)
    
    return {
        'predictions': predictions_df,
        'targets': targets_df,
        'inference_times': inference_times,
        'stability_metrics': stability_metrics,
        'num_images': len(dataset)
    }

def compute_coco_metrics(predictions_df, targets_df, dataset, logger):
    """Compute COCO metrics using pycocotools."""
    
    # Convert to COCO format
    coco_gt = COCO()
    coco_gt.dataset = dataset.coco.dataset
    coco_gt.createIndex()
    
    # Create predictions in COCO format
    coco_dt = coco_gt.loadRes(predictions_df.to_dict('records'))
    
    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'mAP@0.5:0.95': float(coco_eval.stats[0]),
        'mAP@0.5': float(coco_eval.stats[1]),
        'mAP@0.75': float(coco_eval.stats[2]),
        'mAP_small': float(coco_eval.stats[3]),
        'mAP_medium': float(coco_eval.stats[4]),
        'mAP_large': float(coco_eval.stats[5]),
        'AR_max1': float(coco_eval.stats[6]),
        'AR_max10': float(coco_eval.stats[7]),
        'AR_max100': float(coco_eval.stats[8]),
        'AR_small': float(coco_eval.stats[9]),
        'AR_medium': float(coco_eval.stats[10]),
        'AR_large': float(coco_eval.stats[11])
    }
    
    return metrics

def compute_performance_metrics(inference_times, stability_metrics, logger):
    """Compute performance metrics."""
    
    # Inference speed metrics
    inference_times = np.array(inference_times)
    
    speed_metrics = {
        'mean_inference_time_ms': float(np.mean(inference_times) * 1000),
        'std_inference_time_ms': float(np.std(inference_times) * 1000),
        'p95_inference_time_ms': float(np.percentile(inference_times, 95) * 1000),
        'p99_inference_time_ms': float(np.percentile(inference_times, 99) * 1000),
        'fps': float(1.0 / np.mean(inference_times)),
        'min_fps': float(1.0 / np.max(inference_times)),
        'max_fps': float(1.0 / np.min(inference_times))
    }
    
    # Stability metrics
    stability_results = {}
    if stability_metrics:
        # Aggregate stability metrics
        for key in stability_metrics[0].keys():
            values = [m[key] for m in stability_metrics if key in m]
            if values:
                stability_results[f'stability_{key}_mean'] = float(np.mean(values))
                stability_results[f'stability_{key}_std'] = float(np.std(values))
    
    return {**speed_metrics, **stability_results}

def export_results(results, output_dir, logger):
    """Export evaluation results to files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    results['predictions'].to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save metrics
    metrics = {
        'detection_metrics': results['detection_metrics'],
        'performance_metrics': results['performance_metrics'],
        'summary': {
            'num_images_evaluated': results['num_images'],
            'num_predictions': len(results['predictions']),
            'num_targets': len(results['targets']),
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed report
    report = f"""
    ========================================
    EVALUATION REPORT
    ========================================
    
    Dataset: {results.get('dataset_name', 'Unknown')}
    Model: {results.get('model_name', 'Unknown')}
    Images Evaluated: {results['num_images']}
    
    DETECTION METRICS:
    ------------------
    mAP@0.5:0.95: {results['detection_metrics']['mAP@0.5:0.95']:.4f}
    mAP@0.5: {results['detection_metrics']['mAP@0.5']:.4f}
    mAP@0.75: {results['detection_metrics']['mAP@0.75']:.4f}
    
    PERFORMANCE METRICS:
    --------------------
    Mean Inference Time: {results['performance_metrics']['mean_inference_time_ms']:.2f} ms
    FPS: {results['performance_metrics']['fps']:.2f}
    P95 Inference Time: {results['performance_metrics']['p95_inference_time_ms']:.2f} ms
    
    STABILITY METRICS:
    ------------------
    """
    
    for key, value in results['performance_metrics'].items():
        if key.startswith('stability_'):
            report += f"{key}: {value:.4f}\n"
    
    with open(output_dir / 'report.txt', 'w') as f:
        f.write(report)
    
    logger.info(f"Results exported to {output_dir}")

def main(args):
    """Main evaluation function."""
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = ModelConfig(**config_dict.get('model', {}))
    inference_config = InferenceConfig(**config_dict.get('inference', {}))
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model = HybridVisionSystem(
        config=model_config,
        num_classes=args.num_classes or model_config.num_classes,
        use_vit=model_config.use_vit,
        use_rag=model_config.use_rag
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(inference_config.device)
    model.eval()
    
    # Create dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    
    val_transforms = create_val_transforms(
        img_size=inference_config.image_size
    )
    
    dataset = COCOVisionDataset(
        root=args.data_dir,
        annotation_file=args.annotations,
        transforms=val_transforms,
        cache=False
    )
    
    # Run evaluation
    logger.info("Starting evaluation")
    
    results = evaluate_model(model, dataset, inference_config, logger)
    
    # Compute metrics
    logger.info("Computing COCO metrics")
    detection_metrics = compute_coco_metrics(
        results['predictions'],
        results['targets'],
        dataset,
        logger
    )
    
    logger.info("Computing performance metrics")
    performance_metrics = compute_performance_metrics(
        results['inference_times'],
        results['stability_metrics'],
        logger
    )
    
    # Combine results
    full_results = {
        'predictions': results['predictions'],
        'targets': results['targets'],
        'detection_metrics': detection_metrics,
        'performance_metrics': performance_metrics,
        'num_images': results['num_images'],
        'model_name': args.model_name or Path(args.model_path).stem,
        'dataset_name': Path(args.data_dir).name
    }
    
    # Export results
    export_results(full_results, args.output_dir, logger)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"mAP@0.5:0.95: {detection_metrics['mAP@0.5:0.95']:.4f}")
    logger.info(f"mAP@0.5: {detection_metrics['mAP@0.5']:.4f}")
    logger.info(f"FPS: {performance_metrics['fps']:.2f}")
    logger.info(f"Mean Inference Time: {performance_metrics['mean_inference_time_ms']:.2f} ms")
    logger.info("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Vision System')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to evaluation data directory')
    
    # Dataset arguments
    parser.add_argument('--annotations', type=str, default='annotations/instances_val2017.json',
                       help='Path to COCO annotations file')
    parser.add_argument('--num-classes', type=int, default=80,
                       help='Number of classes')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/evaluation.yaml',
                       help='Path to configuration file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name for reporting')
    
    # Evaluation options
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    main(args)