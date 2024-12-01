import cv2
import numpy as np
import time
import os

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("C:/pythton/quiz/files/MobileNetSSD_deploy.prototxt",
                               "C:/pythton/quiz/files/MobileNetSSD_deploy.caffemodel")

# Classes to detect (including the vehicles we care about)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor", "truck", "auto"
]

# Initialize video capture for each camera
cameras = [
    cv2.VideoCapture(0),  # Camera 1
    cv2.VideoCapture(1),  # Camera 2
    cv2.VideoCapture(2),  # Camera 3
    cv2.VideoCapture(3)   # Camera 4
]

# Vehicle classes of interest
VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "van", "truck", "auto"}

# Define the reference line position (e.g., 5 meters = 100 pixels, for simulation purposes)
REFERENCE_LINE_Y = 350  # This is the Y-coordinate of the reference line (in pixels)

# Directory to save images
SAVE_DIR = "captured_images/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Traffic light states for each direction
traffic_signals = ['RED', 'RED', 'RED', 'RED']  # Initial states
green_signal_duration = 10  # Duration for green signal in seconds
yellow_signal_duration = 5  # Duration for yellow signal in seconds
green_start_time = None     # Track start time of green signal
yellow_start_time = None    # Track start time of yellow signal
signal_timer = None         # Timer for signals
green_direction = None      # Direction with green signal
yellow_direction = None     # Direction with yellow signal

def detect_and_count_vehicles(frame, net, red_signal):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (250, 250)), 0.007843, (250, 250), 127.5)
    net.setInput(blob)
    detections = net.forward()

    vehicle_count = 0
    vehicles_crossed_line = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            class_name = CLASSES[idx]
            if class_name in VEHICLE_CLASSES:
                vehicle_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw bounding box and label
                label = f"{class_name}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 215, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 205, 255), 2)

                # Check if vehicle crosses the reference line
                if startY < REFERENCE_LINE_Y < endY and red_signal:
                    vehicles_crossed_line.append(class_name)
                    crossing_label = f"{class_name} crossed on RED signal!"
                    cv2.putText(frame, crossing_label, (startX, endY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 205), 2)
                    
                    # Save image with evidence details
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{SAVE_DIR}red_signal_violation_{timestamp}.jpg"
                    
                    # Annotate frame with evidence details
                    evidence_text = f"Violation: {class_name} crossed on RED | Timestamp: {timestamp}"
                    cv2.putText(frame, evidence_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 205), 2)
                    cv2.line(frame, (0, REFERENCE_LINE_Y), (w, REFERENCE_LINE_Y), (0, 0, 205), 2)  # Reference line
                    cv2.imwrite(filename, frame)
                    print(f"Red signal violation recorded: {filename}")

    return frame, vehicle_count, vehicles_crossed_line, w

# Main loop to process frames from each camera
try:
    while True:
        vehicle_counts = []
        vehicles_crossed_all = []

        for i, cam in enumerate(cameras):
            ret, frame = cam.read()
            if not ret:
                print(f"Failed to grab frame from camera {i+1}")
                vehicle_counts.append(0)
                vehicles_crossed_all.append([])
                continue

            # Determine if signal is red based on traffic signal state
            red_signal = traffic_signals[i] == 'RED'
            frame, vehicle_count, vehicles_crossed, w = detect_and_count_vehicles(frame, net, red_signal)
            vehicle_counts.append(vehicle_count)
            vehicles_crossed_all.append(vehicles_crossed)

            # Draw the reference line
            cv2.line(frame, (0, REFERENCE_LINE_Y), (w, REFERENCE_LINE_Y), (0, 0, 205), 2)
            cv2.putText(frame, "Reference Line", (10, REFERENCE_LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 0, 0), 2)

            # Display frame
            cv2.imshow(f"Camera {i+1}", frame)

        # Find the direction with the highest and second-highest vehicle counts
        sorted_indices = sorted(range(len(vehicle_counts)), key=lambda x: vehicle_counts[x], reverse=True)
        max_index = sorted_indices[0]
        second_max_index = sorted_indices[1] if len(sorted_indices) > 1 else None

        # Check if it's time to change the signals
        current_time = time.time()
        if green_start_time is None or current_time - green_start_time >= green_signal_duration + yellow_signal_duration:
            # Set signals: green for max_index, yellow for second_max_index, red for others
            traffic_signals = ['RED'] * 4
            traffic_signals[max_index] = 'GREEN'
            green_direction = max_index
            if second_max_index is not None:
                traffic_signals[second_max_index] = 'YELLOW'
                yellow_direction = second_max_index
            green_start_time = current_time
            signal_timer = current_time

        # Handle yellow signal duration
        if yellow_direction is not None and current_time - signal_timer >= green_signal_duration and current_time - signal_timer < green_signal_duration + yellow_signal_duration:
            traffic_signals[yellow_direction] = 'YELLOW'

        # Reset to red after signal durations have passed
        if current_time - signal_timer >= green_signal_duration + yellow_signal_duration:
            # Reset yellow to red
            if yellow_direction is not None:
                traffic_signals[yellow_direction] = 'RED'
            yellow_direction = None  # Reset yellow direction
            signal_timer = None  # Reset signal timer

        # Display current signal states
        for j, signal in enumerate(traffic_signals):
            print(f"Camera {j+1} signal: {signal}")

        # Print vehicle counts and crossings
        print(f"Vehicle counts from all cameras: {vehicle_counts}")
        for i, vehicles in enumerate(vehicles_crossed_all):
            if vehicles:
                print(f"Vehicles crossing the reference line at Camera {i+1}: {vehicles}")

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()