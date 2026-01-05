#!/usr/bin/env python3
"""
Deployment script for Hybrid Vision System.
Handles Docker building, Kubernetes deployment, and cloud deployment.
"""

import os
import sys
import argparse
import yaml
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import tempfile
import docker
import boto3
import google.auth
from google.cloud import storage, aiplatform
from kubernetes import client, config

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.logging import setup_logging, get_logger

class DeploymentManager:
    """Manage deployment of the vision system."""
    
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.deployment_config = self.config.get('deployment', {})
        
        # Set up paths
        self.project_root = Path(__file__).parent.parent
        self.docker_dir = self.project_root / 'docker'
        self.kubernetes_dir = self.project_root / 'kubernetes'
        self.models_dir = self.project_root / 'models'
        
        # Create output directory
        self.output_dir = self.project_root / 'deployment'
        self.output_dir.mkdir(exist_ok=True)
    
    def build_docker_image(self, image_tag=None, build_type='inference'):
        """Build Docker image."""
        
        self.logger.info(f"Building Docker image for {build_type}")
        
        if image_tag is None:
            image_tag = f"hybrid-vision-{build_type}:{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Select Dockerfile
        if build_type == 'training':
            dockerfile = self.docker_dir / 'Dockerfile.train'
        else:
            dockerfile = self.docker_dir / 'Dockerfile.inference'
        
        if not dockerfile.exists():
            self.logger.error(f"Dockerfile not found: {dockerfile}")
            return None
        
        # Build command
        cmd = [
            'docker', 'build',
            '-f', str(dockerfile),
            '-t', image_tag,
            '--build-arg', f'MODEL_PATH=/models/vision_model.pt',
            '--build-arg', f'CONFIG_PATH=/configs/deployment.yaml',
            str(self.project_root)
        ]
        
        # Add build args from config
        build_args = self.deployment_config.get('docker', {}).get('build_args', {})
        for key, value in build_args.items():
            cmd.extend(['--build-arg', f'{key}={value}'])
        
        # Add platform if specified
        if 'platform' in self.deployment_config.get('docker', {}):
            cmd.extend(['--platform', self.deployment_config['docker']['platform']])
        
        # Run build
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"Docker build output:\n{result.stdout}")
            
            if result.stderr:
                self.logger.warning(f"Docker build warnings:\n{result.stderr}")
            
            self.logger.info(f"Docker image built successfully: {image_tag}")
            return image_tag
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker build failed: {e}")
            self.logger.error(f"Error output:\n{e.stderr}")
            return None
    
    def push_docker_image(self, image_tag, registry=None):
        """Push Docker image to registry."""
        
        self.logger.info(f"Pushing Docker image: {image_tag}")
        
        # Determine registry
        if registry is None:
            registry = self.deployment_config.get('docker', {}).get('registry', '')
        
        # Tag image for registry
        if registry and not image_tag.startswith(registry):
            registry_tag = f"{registry}/{image_tag}"
            cmd = ['docker', 'tag', image_tag, registry_tag]
            
            self.logger.info(f"Tagging: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            image_tag = registry_tag
        
        # Push image
        cmd = ['docker', 'push', image_tag]
        self.logger.info(f"Pushing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"Docker push output:\n{result.stdout}")
            
            self.logger.info(f"Docker image pushed successfully: {image_tag}")
            return image_tag
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker push failed: {e}")
            self.logger.error(f"Error output:\n{e.stderr}")
            return None
    
    def deploy_kubernetes(self, image_tag, namespace='robot-vision'):
        """Deploy to Kubernetes cluster."""
        
        self.logger.info(f"Deploying to Kubernetes with image: {image_tag}")
        
        # Load kubeconfig
        try:
            config.load_kube_config()
        except:
            self.logger.warning("Could not load kubeconfig, trying in-cluster config")
            config.load_incluster_config()
        
        # Create Kubernetes API clients
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        networking_v1 = client.NetworkingV1Api()
        
        # Create namespace if it doesn't exist
        try:
            core_v1.create_namespace(
                client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
            )
            self.logger.info(f"Created namespace: {namespace}")
        except client.exceptions.ApiException as e:
            if e.status != 409:  # Already exists
                raise
        
        # Update deployment YAML with image tag
        deployment_path = self.kubernetes_dir / 'deployment.yaml'
        
        with open(deployment_path, 'r') as f:
            deployment_yaml = yaml.safe_load(f)
        
        # Update image
        deployment_yaml['spec']['template']['spec']['containers'][0]['image'] = image_tag
        
        # Create or update deployment
        try:
            # Check if deployment exists
            existing_deployment = apps_v1.read_namespaced_deployment(
                name='hybrid-vision-inference',
                namespace=namespace
            )
            
            # Update existing deployment
            existing_deployment.spec.template.spec.containers[0].image = image_tag
            apps_v1.replace_namespaced_deployment(
                name='hybrid-vision-inference',
                namespace=namespace,
                body=existing_deployment
            )
            self.logger.info("Updated existing deployment")
            
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Create new deployment
                apps_v1.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment_yaml
                )
                self.logger.info("Created new deployment")
            else:
                raise
        
        # Create service if it doesn't exist
        service_path = self.kubernetes_dir / 'service.yaml'
        
        with open(service_path, 'r') as f:
            service_yaml = yaml.safe_load(f)
        
        try:
            core_v1.read_namespaced_service(
                name='vision-inference-service',
                namespace=namespace
            )
            self.logger.info("Service already exists")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                core_v1.create_namespaced_service(
                    namespace=namespace,
                    body=service_yaml
                )
                self.logger.info("Created service")
            else:
                raise
        
        # Create configmap
        configmap_path = self.kubernetes_dir / 'configmap.yaml'
        
        if configmap_path.exists():
            with open(configmap_path, 'r') as f:
                configmap_yaml = yaml.safe_load(f)
            
            try:
                core_v1.create_namespaced_config_map(
                    namespace=namespace,
                    body=configmap_yaml
                )
                self.logger.info("Created configmap")
            except client.exceptions.ApiException as e:
                if e.status == 409:
                    self.logger.info("Configmap already exists")
                else:
                    raise
        
        self.logger.info("Kubernetes deployment completed")
        
        # Get service endpoint
        try:
            service = core_v1.read_namespaced_service(
                name='vision-inference-service',
                namespace=namespace
            )
            
            if service.spec.type == 'LoadBalancer':
                for ingress in service.status.load_balancer.ingress or []:
                    if ingress.ip:
                        self.logger.info(f"Service endpoint: http://{ingress.ip}")
                    elif ingress.hostname:
                        self.logger.info(f"Service endpoint: http://{ingress.hostname}")
            else:
                self.logger.info(f"Service type: {service.spec.type}")
                
        except Exception as e:
            self.logger.warning(f"Could not get service endpoint: {e}")
    
    def deploy_aws_sagemaker(self, model_path, instance_type='ml.g4dn.xlarge'):
        """Deploy to AWS SageMaker."""
        
        self.logger.info("Deploying to AWS SageMaker")
        
        # Check AWS credentials
        session = boto3.Session()
        if session.get_credentials() is None:
            self.logger.error("AWS credentials not found")
            return False
        
        # Create temporary directory for deployment artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Copy model and code
            shutil.copy(model_path, tmpdir / 'model.tar.gz')
            
            # Create inference code
            inference_code = tmpdir / 'code'
            inference_code.mkdir()
            
            # Copy inference scripts
            shutil.copy(self.project_root / 'src' / 'deployment' / 'api_server.py',
                       inference_code / 'inference.py')
            shutil.copy(self.project_root / 'src' / 'deployment' / 'requirements.txt',
                       inference_code / 'requirements.txt')
            
            # Create Dockerfile for SageMaker
            dockerfile_content = f"""
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV SAGEMAKER_PROGRAM inference.py
ENV MODEL_PATH /opt/ml/model/model.tar.gz

CMD ["python", "inference.py"]
"""
            
            with open(inference_code / 'Dockerfile', 'w') as f:
                f.write(dockerfile_content)
            
            # Create model.tar.gz
            model_archive = tmpdir / 'model.tar.gz'
            subprocess.run(['tar', '-czf', str(model_archive), '-C', str(tmpdir), 'model.tar.gz', 'code'],
                         check=True)
            
            # Upload to S3
            s3_client = boto3.client('s3')
            bucket = self.deployment_config.get('aws', {}).get('s3_bucket', 'hybrid-vision-models')
            
            # Create bucket if it doesn't exist
            try:
                s3_client.head_bucket(Bucket=bucket)
            except:
                s3_client.create_bucket(Bucket=bucket)
            
            # Upload model
            s3_key = f"models/{datetime.now().strftime('%Y%m%d')}/model.tar.gz"
            s3_client.upload_file(str(model_archive), bucket, s3_key)
            
            self.logger.info(f"Model uploaded to s3://{bucket}/{s3_key}")
            
            # Create SageMaker model
            sagemaker_client = boto3.client('sagemaker')
            
            model_name = f"hybrid-vision-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Create model
            sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-gpu-py310',
                    'ModelDataUrl': f"s3://{bucket}/{s3_key}",
                    'Environment': {
                        'MODEL_PATH': '/opt/ml/model/model.tar.gz'
                    }
                },
                ExecutionRoleArn=self.deployment_config.get('aws', {}).get('role_arn')
            )
            
            self.logger.info(f"SageMaker model created: {model_name}")
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config"
            
            sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }]
            )
            
            # Create endpoint
            endpoint_name = f"hybrid-vision-endpoint"
            
            try:
                sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
                self.logger.info(f"SageMaker endpoint created: {endpoint_name}")
                
            except sagemaker_client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == 'ValidationException':
                    # Update existing endpoint
                    sagemaker_client.update_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=endpoint_config_name
                    )
                    self.logger.info(f"SageMaker endpoint updated: {endpoint_name}")
                else:
                    raise
        
        return True
    
    def deploy_gcp_vertex_ai(self, model_path, machine_type='n1-standard-4'):
        """Deploy to Google Cloud Vertex AI."""
        
        self.logger.info("Deploying to Google Cloud Vertex AI")
        
        try:
            # Authenticate
            credentials, project = google.auth.default()
            
            # Initialize Vertex AI
            aiplatform.init(project=project, location='us-central1')
            
            # Upload model to GCS
            storage_client = storage.Client()
            bucket_name = self.deployment_config.get('gcp', {}).get('bucket', 'hybrid-vision-models')
            
            bucket = storage_client.bucket(bucket_name)
            if not bucket.exists():
                bucket = storage_client.create_bucket(bucket_name, location='us-central1')
            
            # Upload model
            gcs_path = f"gs://{bucket_name}/models/{datetime.now().strftime('%Y%m%d')}/model.tar.gz"
            blob = bucket.blob(f"models/{datetime.now().strftime('%Y%m%d')}/model.tar.gz")
            blob.upload_from_filename(model_path)
            
            self.logger.info(f"Model uploaded to {gcs_path}")
            
            # Create Vertex AI model
            model = aiplatform.Model.upload(
                display_name=f"hybrid-vision-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                artifact_uri=f"gs://{bucket_name}/models/{datetime.now().strftime('%Y%m%d')}",
                serving_container_image_uri='us-docker.pkg.dev/vertex-ai/pytorch/pytorch:2.0.0-gpu',
                serving_container_ports=[8080],
                serving_container_environment_variables={
                    'MODEL_PATH': '/opt/ml/model/model.tar.gz'
                }
            )
            
            self.logger.info(f"Vertex AI model created: {model.display_name}")
            
            # Deploy model to endpoint
            endpoint = model.deploy(
                deployed_model_display_name=model.display_name,
                machine_type=machine_type,
                accelerator_type='NVIDIA_TESLA_T4',
                accelerator_count=1,
                min_replica_count=1,
                max_replica_count=3,
                sync=True
            )
            
            self.logger.info(f"Model deployed to endpoint: {endpoint.display_name}")
            self.logger.info(f"Endpoint ID: {endpoint.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"GCP deployment failed: {e}")
            return False
    
    def deploy_azure_ml(self, model_path, compute_target='gpu-cluster'):
        """Deploy to Azure Machine Learning."""
        
        self.logger.info("Deploying to Azure Machine Learning")
        
        try:
            from azure.identity import DefaultAzureCredential
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import Model, Environment, CodeConfiguration
            from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
            
            # Authenticate
            credential = DefaultAzureCredential()
            
            # Get workspace
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.deployment_config.get('azure', {}).get('subscription_id'),
                resource_group_name=self.deployment_config.get('azure', {}).get('resource_group'),
                workspace_name=self.deployment_config.get('azure', {}).get('workspace')
            )
            
            # Register model
            model = Model(
                path=model_path,
                name=f"hybrid-vision-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                description="Hybrid Vision System for robotic perception"
            )
            
            registered_model = ml_client.models.create_or_update(model)
            self.logger.info(f"Model registered: {registered_model.name}")
            
            # Create environment
            env = Environment(
                name="hybrid-vision-env",
                description="Environment for Hybrid Vision System",
                conda_file=str(self.project_root / 'environment.yml'),
                image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:latest"
            )
            
            # Create online endpoint
            endpoint = ManagedOnlineEndpoint(
                name="hybrid-vision-endpoint",
                description="Online endpoint for Hybrid Vision System",
                auth_mode="key"
            )
            
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            
            # Create deployment
            deployment = ManagedOnlineDeployment(
                name="blue",
                endpoint_name=endpoint.name,
                model=registered_model.name,
                environment=env,
                code_configuration=CodeConfiguration(
                    code=str(self.project_root / 'src' / 'deployment'),
                    scoring_script="api_server.py"
                ),
                instance_type="Standard_NC6s_v3",
                instance_count=1
            )
            
            ml_client.online_deployments.begin_create_or_update(deployment).result()
            
            # Update endpoint traffic
            endpoint.traffic = {"blue": 100}
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            
            # Get endpoint details
            endpoint = ml_client.online_endpoints.get(name=endpoint.name)
            self.logger.info(f"Endpoint URI: {endpoint.scoring_uri}")
            
            return True
            
        except ImportError:
            self.logger.error("Azure ML SDK not installed. Install with: pip install azure-ai-ml")
            return False
        except Exception as e:
            self.logger.error(f"Azure deployment failed: {e}")
            return False
    
    def deploy_edge_device(self, model_path, device_ip, device_user='ubuntu'):
        """Deploy to edge device (Jetson, Raspberry Pi, etc.)."""
        
        self.logger.info(f"Deploying to edge device: {device_ip}")
        
        # Create deployment package
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create deployment directory structure
            deploy_dir = tmpdir / 'deployment'
            deploy_dir.mkdir()
            
            # Copy model
            shutil.copy(model_path, deploy_dir / 'vision_model.pt')
            
            # Copy configuration
            shutil.copy(self.config_path, deploy_dir / 'config.yaml')
            
            # Copy inference code
            shutil.copytree(self.project_root / 'src' / 'inference',
                          deploy_dir / 'inference')
            shutil.copytree(self.project_root / 'src' / 'deployment',
                          deploy_dir / 'deployment')
            shutil.copytree(self.project_root / 'src' / 'utils',
                          deploy_dir / 'utils')
            
            # Create requirements file for edge device
            requirements = [
                'torch>=2.0.0',
                'torchvision>=0.15.0',
                'numpy>=1.24.0',
                'opencv-python>=4.7.0',
                'pyyaml>=6.0'
            ]
            
            with open(deploy_dir / 'requirements.txt', 'w') as f:
                f.write('\n'.join(requirements))
            
            # Create startup script
            startup_script = """#!/bin/bash
cd /opt/hybrid-vision
pip install -r requirements.txt
python -m deployment.api_server --model vision_model.pt --host 0.0.0.0 --port 8000
"""
            
            with open(deploy_dir / 'start.sh', 'w') as f:
                f.write(startup_script)
            
            os.chmod(deploy_dir / 'start.sh', 0o755)
            
            # Create tar archive
            archive_path = tmpdir / 'deployment.tar.gz'
            subprocess.run(['tar', '-czf', str(archive_path), '-C', str(deploy_dir), '.'],
                         check=True)
            
            # Copy to edge device
            scp_cmd = ['scp', str(archive_path), f'{device_user}@{device_ip}:/tmp/']
            
            self.logger.info(f"Copying deployment package: {' '.join(scp_cmd)}")
            subprocess.run(scp_cmd, check=True)
            
            # SSH commands to deploy
            ssh_commands = [
                f'sudo mkdir -p /opt/hybrid-vision',
                f'sudo tar -xzf /tmp/deployment.tar.gz -C /opt/hybrid-vision',
                f'sudo chown -R {device_user}:{device_user} /opt/hybrid-vision',
                f'cd /opt/hybrid-vision && pip install -r requirements.txt',
                f'sudo systemctl stop hybrid-vision 2>/dev/null || true',
                f'sudo tee /etc/systemd/system/hybrid-vision.service << EOF\n'
                f'[Unit]\n'
                f'Description=Hybrid Vision System\n'
                f'After=network.target\n\n'
                f'[Service]\n'
                f'Type=simple\n'
                f'User={device_user}\n'
                f'WorkingDirectory=/opt/hybrid-vision\n'
                f'ExecStart=/bin/bash /opt/hybrid-vision/start.sh\n'
                f'Restart=always\n'
                f'RestartSec=10\n\n'
                f'[Install]\n'
                f'WantedBy=multi-user.target\n'
                f'EOF\n',
                f'sudo systemctl daemon-reload',
                f'sudo systemctl enable hybrid-vision',
                f'sudo systemctl start hybrid-vision',
                f'sleep 2',
                f'sudo systemctl status hybrid-vision --no-pager'
            ]
            
            # Execute SSH commands
            for cmd in ssh_commands:
                ssh_cmd = ['ssh', f'{device_user}@{device_ip}', cmd]
                self.logger.info(f"Executing: {' '.join(ssh_cmd[:2])} ...")
                subprocess.run(ssh_cmd, check=True)
        
        self.logger.info(f"Deployment to edge device {device_ip} completed")
        self.logger.info(f"Service available at: http://{device_ip}:8000")
        
        return True
    
    def run(self, target, **kwargs):
        """Run deployment to specified target."""
        
        self.logger.info(f"Starting deployment to {target}")
        
        # Get model path
        model_path = kwargs.get('model_path', self.models_dir / 'vision_model.pt')
        
        if not Path(model_path).exists():
            self.logger.error(f"Model not found: {model_path}")
            return False
        
        # Run deployment based on target
        if target == 'docker':
            image_tag = self.build_docker_image(
                image_tag=kwargs.get('image_tag'),
                build_type=kwargs.get('build_type', 'inference')
            )
            
            if image_tag and kwargs.get('push', False):
                self.push_docker_image(image_tag, kwargs.get('registry'))
            
            return image_tag is not None
        
        elif target == 'kubernetes':
            image_tag = kwargs.get('image_tag')
            if not image_tag:
                # Build image first
                image_tag = self.build_docker_image(build_type='inference')
            
            if image_tag:
                self.deploy_kubernetes(
                    image_tag=image_tag,
                    namespace=kwargs.get('namespace', 'robot-vision')
                )
                return True
            return False
        
        elif target == 'aws':
            return self.deploy_aws_sagemaker(
                model_path=model_path,
                instance_type=kwargs.get('instance_type', 'ml.g4dn.xlarge')
            )
        
        elif target == 'gcp':
            return self.deploy_gcp_vertex_ai(
                model_path=model_path,
                machine_type=kwargs.get('machine_type', 'n1-standard-4')
            )
        
        elif target == 'azure':
            return self.deploy_azure_ml(
                model_path=model_path,
                compute_target=kwargs.get('compute_target', 'gpu-cluster')
            )
        
        elif target == 'edge':
            return self.deploy_edge_device(
                model_path=model_path,
                device_ip=kwargs.get('device_ip'),
                device_user=kwargs.get('device_user', 'ubuntu')
            )
        
        else:
            self.logger.error(f"Unknown deployment target: {target}")
            return False

def main(args):
    """Main deployment function."""
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Create deployment manager
    manager = DeploymentManager(args.config, logger)
    
    # Run deployment
    success = manager.run(
        target=args.target,
        model_path=args.model_path,
        image_tag=args.image_tag,
        build_type=args.build_type,
        push=args.push,
        registry=args.registry,
        namespace=args.namespace,
        device_ip=args.device_ip,
        device_user=args.device_user
    )
    
    if success:
        logger.info(f"Deployment to {args.target} completed successfully")
    else:
        logger.error(f"Deployment to {args.target} failed")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy Hybrid Vision System')
    
    # Deployment target
    parser.add_argument('--target', type=str, required=True,
                       choices=['docker', 'kubernetes', 'aws', 'gcp', 'azure', 'edge'],
                       help='Deployment target')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/deployment.yaml',
                       help='Path to deployment configuration file')
    
    # Model
    parser.add_argument('--model-path', type=str, default='models/vision_model.pt',
                       help='Path to trained model')
    
    # Docker/Kubernetes options
    parser.add_argument('--image-tag', type=str, default=None,
                       help='Docker image tag')
    parser.add_argument('--build-type', type=str, default='inference',
                       choices=['training', 'inference'],
                       help='Docker build type')
    parser.add_argument('--push', action='store_true',
                       help='Push Docker image to registry')
    parser.add_argument('--registry', type=str, default=None,
                       help='Docker registry URL')
    parser.add_argument('--namespace', type=str, default='robot-vision',
                       help='Kubernetes namespace')
    
    # Edge deployment options
    parser.add_argument('--device-ip', type=str, default=None,
                       help='Edge device IP address')
    parser.add_argument('--device-user', type=str, default='ubuntu',
                       help='Edge device username')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target == 'edge' and not args.device_ip:
        parser.error("--device-ip is required for edge deployment")
    
    main(args)