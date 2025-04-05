# VisionDepthNet
3D distance estimation using monocular vision

# Depth Estimation Docker

## Quick Start
```bash
# Build the image
docker build -t depth-estimator .

# Run with default image
docker run -it --rm depth-estimator

# Run with your own image (mount volume)
docker run -it --rm \
  -v $(pwd)/images:/app/images \
  depth-estimator \
  python depth_estimation.py --image images/yourphoto.jpg
```

## Requirements
- Docker installed
- NVIDIA Container Toolkit (for GPU support)