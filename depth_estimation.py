import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. Calibration parameters (feel free to adjust)
SCALE_FACTOR = 600_000
CLASSES = [0, 1, 2]     # 0 = person, 1 = bike, 2 = car

FRAME_SIZE = (384, 640)

# 2. Detection and segmentation function
def detect_and_segment(image_path, conf_threshold=0.5):
    model = YOLO("yolov8s-seg.pt")  # Model with segmentation
    results = model.predict(image_path, conf=conf_threshold, classes=CLASSES)
    
    detections = []
    for result in results:
        for i, box in enumerate(result.boxes):
            # Bounding Box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Segmentation mask
            mask = result.masks[i].data[0].cpu().numpy()
            mask = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "mask": mask,
                "class": result.names[int(box.cls[0])],
                "confidence": box.conf[0].item()
            })
    
    return detections, results[0].orig_img

# 3. Depth model initialization
def setup_depth_estimator():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    model.eval()
    
    transforms = Compose([
        Resize(FRAME_SIZE),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return model, transforms

# 4. Depth estimation
def estimate_depth(image, model, transforms):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transforms(img_pil).unsqueeze(0)
    
    with torch.no_grad():
        disparity = model(input_tensor)
    
    depth_map = disparity.squeeze().cpu().numpy()

    return depth_map  # Immediate calibration

# 5. Distance calculation
def calculate_distances(detections, depth_map, original_image_shape):
    # Resize depth map to original image size
    depth_map = cv2.resize(
        depth_map, 
        (original_image_shape[1], original_image_shape[0]),  # (width, height)
        interpolation=cv2.INTER_CUBIC
    )
    
    for det in detections:
        mask = det["mask"] > 0.5
        object_depths = depth_map[mask]

        if object_depths.size == 0:
            det["distance"] = None
            continue
        # Find minimum depth
        max_depth = np.max(object_depths)
        average_depth = np.mean(object_depths)
        det["distance"] = average_depth
        
        # Find position of closest point
        y, x = np.where((depth_map == max_depth) & mask)
        det["closest_point"] = (x[0], y[0]) if len(x) > 0 else None
    
    return detections

# 5.1 Convert depth map to meters
def convert_to_meters(depth_map):

    depth_map_meters = SCALE_FACTOR / (depth_map * depth_map)
    return depth_map_meters

# 6. Visualization
def visualize(image, detections, depth_map, save_path="outputx.jpg"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Image with masks and bounding boxes
    display_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    for det in detections:
        if det["distance"] is None:
            continue
        
        # Segmentation mask overlay
        mask = det["mask"] > 0.5
        overlay = display_image.copy()
        overlay[mask] = [0, 255, 255]  # Cyan color for masks
        cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0, display_image)
        
        # Bounding box and label
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0,255,0), 2)
        print(f"{det['class']} {det['distance']:.1f}m")
        cv2.putText(display_image, f"{det['class']} {det['distance']:.1f}m", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Closest point marker
        if det["closest_point"]:
            x, y = det["closest_point"]
            cv2.circle(display_image, (x, y), 8, (255,0,0), -1)
    
    if save_path:
        cv2.imwrite(save_path, display_image)

    else:

        ax1.imshow(display_image)
        ax1.set_title("Detections with masks and distances")
    
        # Depth map display
        ax2.imshow(depth_map, cmap="plasma")
        ax2.set_title("Depth map")
        
        plt.show()

# 7. Main pipeline
if __name__ == "__main__":
    # Step 1: Detection + Segmentation
    detections, image = detect_and_segment("testimage.jpg")
    original_shape = image.shape[:2]
    
    # Step 2: Depth model initialization
    depth_model, depth_transforms = setup_depth_estimator()
    
    # Step 3: Depth estimation
    depth_map = estimate_depth(image, depth_model, depth_transforms)
    depth_map_meters = convert_to_meters(depth_map)
    
    # Step 4: Distance calculations
    detections = calculate_distances(detections, depth_map_meters, original_shape)
    
    # Step 5: Visualization
    visualize(image, detections, depth_map_meters)