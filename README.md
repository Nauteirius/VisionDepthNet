# VisionDepthNet

The main lesson's goal is to introduce methos of 3D distance estimation using monocular vision. In particular the focus is on recognition of pedestrians using dashcam-like input to assist autonomous vehicles in making velocity reduction decisions.

The implemented idea is effectively a pipeline that utilizes YOLO and MiDaS solutions to:
- first: detect and segment parts of image that include pedestrians
- second: create a depth map and extract distance in meters between photo taker and subsequent people in the picture

Repository consists of following files:
- depth_estimation.py - proposed pipeline to one-shot estimate distance to pedestrians from the camera perspective
- car_copilot.py - imitation of real-life usage of depth estimation to estimate braking distance length
- Dockerfile - configuration file to run the depth_estimation.py in dockerized way
- requirements.txt - set of module requirements
- testjpg.jpg - sample image (source)[]
- testvideo.mp4 - sample video (source)[https://videos.pexels.com/video-files/5921059/5921059-uhd_3840_2160_30fps.mp4]


# Depth Estimation Docker instruction

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

You can refrain from using Docker. Install requirements.txt and run depth_estimation or car_copilot e.g.:
```bash
pip install -r requirements.txt

python depth_estimation.py --image images/yourphoto.jpg

python car_copilot.py
```

For lightweight version of depth_estimation, replace **DPT_Large** with **MiDaS_small**.

## Requirements
- packages listed in requirements.txt
- Docker installed (optional)
- NVIDIA Container Toolkit (for GPU support)