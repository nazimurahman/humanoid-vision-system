"""
Configuration module for Humanoid Vision System.
Provides easy access to all configuration classes.
"""

from .base_config import BaseConfig, DeviceType, PrecisionType
from .model_config import (
    ModelConfig, MHCConfig, BackboneConfig, ViTConfig,
    FusionConfig, DetectionHeadConfig, RAGConfig,
    BackboneType, FusionType, HeadType, ActivationType
)
from .training_config import (
    TrainingConfig, DatasetConfig, AugmentationConfig,
    OptimizerConfig, SchedulerConfig, LossConfig,
    OptimizerType, SchedulerType, LossType,
    DatasetType, AugmentationType
)
from .inference_config import (
    InferenceConfig, PreprocessingConfig, PostprocessingConfig,
    VisualizationConfig, APIConfig, GRPCConfig, PerformanceConfig,
    InferenceEngine, InputFormat, OutputFormat, VisualizationType
)

__all__ = [
    # Base config
    'BaseConfig', 'DeviceType', 'PrecisionType',
    
    # Model config
    'ModelConfig', 'MHCConfig', 'BackboneConfig', 'ViTConfig',
    'FusionConfig', 'DetectionHeadConfig', 'RAGConfig',
    'BackboneType', 'FusionType', 'HeadType', 'ActivationType',
    
    # Training config
    'TrainingConfig', 'DatasetConfig', 'AugmentationConfig',
    'OptimizerConfig', 'SchedulerConfig', 'LossConfig',
    'OptimizerType', 'SchedulerType', 'LossType',
    'DatasetType', 'AugmentationType',
    
    # Inference config
    'InferenceConfig', 'PreprocessingConfig', 'PostprocessingConfig',
    'VisualizationConfig', 'APIConfig', 'GRPCConfig', 'PerformanceConfig',
    'InferenceEngine', 'InputFormat', 'OutputFormat', 'VisualizationType',
]

def load_config(config_path: str, config_type: str = "model") -> BaseConfig:
    """
    Load configuration from file with automatic type detection.
    
    Args:
        config_path: Path to configuration file
        config_type: Type of config ('model', 'training', 'inference', 'base')
    
    Returns:
        Loaded configuration object
    """
    import yaml
    import json
    from pathlib import Path
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load raw config data
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            data = json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Determine config type from data if not specified
    if config_type == "auto":
        if 'model_name' in data:
            config_type = "model"
        elif 'num_epochs' in data:
            config_type = "training"
        elif 'model_path' in data:
            config_type = "inference"
        else:
            config_type = "base"
    
    # Create appropriate config object
    if config_type == "model":
        return ModelConfig(**data)
    elif config_type == "training":
        return TrainingConfig(**data)
    elif config_type == "inference":
        return InferenceConfig(**data)
    elif config_type == "base":
        return BaseConfig(**data)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

def create_default_configs(output_dir: str = "configs"):
    """
    Create default configuration files.
    
    Args:
        output_dir: Directory to save config files
    """
    import os
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default configs
    configs = {
        'base': BaseConfig(),
        'model': ModelConfig(),
        'training': TrainingConfig(),
        'inference': InferenceConfig(),
    }
    
    # Save each config
    for name, config in configs.items():
        # Save as YAML
        yaml_path = output_dir / f"{name}_config.yaml"
        config.save(yaml_path)
        print(f"Saved {name} config to: {yaml_path}")
        
        # Save as JSON
        json_path = output_dir / f"{name}_config.json"
        config.save(json_path)
        print(f"Saved {name} config to: {json_path}")
    
    print(f"\nAll default configs saved to: {output_dir.absolute()}")

def merge_configs(base_config: BaseConfig, override_config: dict) -> BaseConfig:
    """
    Merge configuration with overrides.
    
    Args:
        base_config: Base configuration object
        override_config: Dictionary of overrides
    
    Returns:
        Merged configuration object
    """
    import copy
    
    # Create a deep copy of the base config
    merged_config = copy.deepcopy(base_config)
    
    # Convert to dict for easier merging
    merged_dict = merged_config.to_dict()
    
    # Recursively update with overrides
    def recursive_update(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    recursive_update(merged_dict, override_config)
    
    # Convert back to appropriate config type
    config_class = type(base_config)
    return config_class(**merged_dict)

# Example usage
if __name__ == "__main__":
    # Create default configs
    create_default_configs()
    
    # Example: Load and display model config
    config = ModelConfig()
    config.display_detailed()
    
    # Example: Save config
    config.save("model_config.yaml")
    
    # Example: Load config
    loaded_config = load_config("model_config.yaml", "model")
    print("\nLoaded config:")
    print(loaded_config.to_json())