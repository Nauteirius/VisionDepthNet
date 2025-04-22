# VisionDepthNet

The main lesson's goal is to introduce methos of 3D distance estimation using monocular vision. In particular the focus is on recognition of pedestrians using dashcam-like input to assist autonomous vehicles in making velocity reduction decisions.

[Lesson slides](https://docs.google.com/presentation/d/1YlnpXRuLLBvxOmRoW507-AbpEdk_sHhxOFYxKhZmyWk/edit?usp=sharing)

The implemented idea is effectively a pipeline that utilizes YOLO and MiDaS solutions to:

- first: detect and segment parts of image that include pedestrians
- second: create a depth map and extract distance in meters between photo taker and subsequent people in the picture

Repository consists of following files:

- depth_estimation.py - proposed pipeline to one-shot estimate distance to pedestrians from the camera perspective
- car_copilot.py - imitation of real-life usage of depth estimation to estimate braking distance length
- requirements.txt - set of module requirements
- testimage.jpg - sample image [source](https://stock.adobe.com/search?k=walking+in+a+city&asset_id=265067956)
- testvideo.mp4 - sample video [source](https://videos.pexels.com/video-files/5921059/5921059-uhd_3840_2160_30fps.mp4)

## Quick Start

Install requirements.txt and run depth_estimation or car_copilot e.g.:

```bash
pip install -r requirements.txt

python depth_estimation.py

python car_copilot.py
```

You are free to use any other package management tool than **pip**.

For lightweight version of depth_estimation, replace **DPT_Large** with **MiDaS_small**.

## Requirements

- packages listed in requirements.txt
- NVIDIA Container Toolkit (optional, for GPU support)
