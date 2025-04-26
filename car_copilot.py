import os
import random
from depth_estimation import *

FPS = 30
FRAME_SIZE = (640, 448)

def road_to_stop(velocity, deceleration=9.8, tire_friction=0.7):   
    effective_deceleration = deceleration * tire_friction

    return (velocity ** 2) / (2 * effective_deceleration)

def press_brake():
    print("Pressing the brake...")

def process_image(frame):
    frame = cv2.resize(frame, FRAME_SIZE)
    detections, image = detect_and_segment(frame)
    original_shape = image.shape[:2]
    
    depth_model, depth_transforms = setup_depth_estimator()
    
    depth_map = estimate_depth(image, depth_model, depth_transforms)
    depth_map_meters = convert_to_meters(depth_map)
    
    detections = calculate_distances(detections, depth_map_meters, original_shape)

    visualize(image, detections, depth_map_meters, save_path="detection.jpg")

    min_distance = float('inf')

    for det in detections:
        if det["class"] == "person" and det["distance"] is not None:
            print(f"Detected {det['class']} at distance: {det['distance']:.2f} m")
            min_distance = min(det["distance"], min_distance)

    if min_distance == float('inf'):
        print("INFO: no person detected.")
    
    return min_distance


def car_copilot(dashcam_video_stream):
    context = 50
    distances = [0.0] * context
    current_index = 0
    frame_count = 0

    cap = cv2.VideoCapture(dashcam_video_stream)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {dashcam_video_stream}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temp_frame_path = "temp.jpg"
        cv2.imwrite(temp_frame_path, frame)

        current_distance = process_image(frame)

        if current_index < context:
            distances[current_index] = current_distance
        else:
            velocity = (current_distance - distances[current_distance]) * FPS
            road_to_stop_distance = road_to_stop(velocity)
            print(f"Velocity: {velocity:.2f} m/s, Road distance to stop: {road_to_stop_distance:.2f} m")

            if road_to_stop_distance > current_distance:
                print("Car is not in a safe distance to stop.")
                press_brake()

        current_index = (current_index + 1) % context    
        frame_count += 1
        print(f"Frame: {frame_count}, Distance: {current_distance:.2f} m")
    cap.release()
    return frame_count

if __name__ == "__main__":
    dashcam_video_stream = "./testvideo.mp4"  # Replace with your movie path
    try:
        car_copilot(dashcam_video_stream)
    except Exception as e:
        print(f"An error occurred: {e}")