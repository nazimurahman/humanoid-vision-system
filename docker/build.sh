#!/bin/bash
# docker/build.sh
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TAG="latest"
TYPE="inference"
PLATFORM="linux/amd64"
PUSH=false
REGISTRY=""
CACHE=true
NO_CACHE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --type)
            TYPE="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --tag TAG        Docker image tag (default: latest)"
            echo "  --type TYPE      Build type: train, inference, all (default: inference)"
            echo "  --platform PLAT  Target platform (default: linux/amd64)"
            echo "  --push           Push to registry after build"
            echo "  --registry REG   Registry URL"
            echo "  --no-cache       Build without cache"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate build type
if [[ ! "$TYPE" =~ ^(train|inference|all)$ ]]; then
    echo -e "${RED}Error: Invalid build type. Use 'train', 'inference', or 'all'${NC}"
    exit 1
fi

# Set build arguments
BUILD_ARGS=""
if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS="--no-cache"
fi

# Function to build image
build_image() {
    local image_type=$1
    local dockerfile="Dockerfile.$image_type"
    local image_name="humanoid-vision-$image_type:$TAG"
    
    if [ -n "$REGISTRY" ]; then
        image_name="$REGISTRY/$image_name"
    fi
    
    echo -e "${YELLOW}Building $image_name from $dockerfile...${NC}"
    
    # Build the image
    docker buildx build \
        --platform $PLATFORM \
        --file docker/$dockerfile \
        --tag $image_name \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        $BUILD_ARGS \
        ../../
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully built $image_name${NC}"
        
        # Push if requested
        if [ "$PUSH" = true ]; then
            echo -e "${YELLOW}Pushing $image_name to registry...${NC}"
            docker push $image_name
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Successfully pushed $image_name${NC}"
            else
                echo -e "${RED}Failed to push $image_name${NC}"
                exit 1
            fi
        fi
    else
        echo -e "${RED}Failed to build $image_name${NC}"
        exit 1
    fi
}

# Create buildx builder if it doesn't exist
if ! docker buildx ls | grep -q vision-builder; then
    echo -e "${YELLOW}Creating buildx builder...${NC}"
    docker buildx create --name vision-builder --use
fi

# Build based on type
case $TYPE in
    train)
        build_image "train"
        ;;
    inference)
        build_image "inference"
        ;;
    all)
        build_image "train"
        build_image "inference"
        ;;
esac

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Available images:"
docker images | grep humanoid-vision

# Create docker-compose override file
if [ "$TYPE" = "inference" ]; then
    cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  inference-api:
    image: ${REGISTRY}humanoid-vision-inference:$TAG
  
  inference-worker:
    image: ${REGISTRY}humanoid-vision-inference:$TAG
EOF
    echo -e "${YELLOW}Created docker-compose.override.yml${NC}"
fi