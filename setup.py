#!/usr/bin/env python
"""Setup script for Humanoid Vision System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = requirements_path.read_text().splitlines() if requirements_path.exists() else []

# Read version from package
version_path = Path(__file__).parent / "src" / "humanoid_vision" / "__init__.py"
version_info = {}
if version_path.exists():
    exec(version_path.read_text(), version_info)
    version = version_info.get("__version__", "0.1.0")
else:
    version = "0.1.0"

setup(
    name="humanoid-vision-system",
    version=version,
    author="Humanoid Vision Team",
    author_email="contact@humanoid-vision.com",
    description="Production-grade hybrid vision system for humanoid robots with manifold-constrained hyper-connections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/humanoid-vision-system",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/humanoid-vision-system/issues",
        "Documentation": "https://github.com/your-org/humanoid-vision-system/docs",
        "Source Code": "https://github.com/your-org/humanoid-vision-system",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12.0",
        "Framework :: FastAPI",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            line.strip()
            for line in (Path(__file__).parent / "requirements-dev.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ],
        "training": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "albumentations>=1.3.0",
        ],
        "deployment": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "gunicorn>=21.2.0",
            "grpcio>=1.57.0",
            "prometheus-client>=0.17.0",
        ],
        "testing": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vision-train=scripts.train:main",
            "vision-infer=scripts.inference:main",
            "vision-api=src.deployment.api_server:main",
            "vision-health=scripts.health_check:main",
        ],
    },
    include_package_data=True,
    package_data={
        "humanoid_vision": [
            "configs/*.yaml",
            "configs/*.yml",
            "models/*.yaml",
        ],
    },
    data_files=[
        ("configs", ["configs/base.yaml", "configs/training.yaml", "configs/inference.yaml"]),
        ("docker", ["docker/Dockerfile.train", "docker/Dockerfile.inference", "docker/docker-compose.yml"]),
        ("kubernetes", ["kubernetes/deployment.yaml", "kubernetes/service.yaml", "kubernetes/hpa.yaml"]),
        ("docs", ["docs/architecture.md", "docs/training_guide.md", "docs/deployment_guide.md"]),
    ],
    scripts=[
        "scripts/train.py",
        "scripts/inference.py",
        "scripts/export_model.py",
        "scripts/benchmark.py",
        "scripts/deploy.py",
    ],
    keywords=[
        "computer-vision",
        "deep-learning",
        "robotics",
        "humanoid-robots",
        "object-detection",
        "vision-transformer",
        "manifold-learning",
        "production-ai",
    ],
    license="Apache License 2.0",
)