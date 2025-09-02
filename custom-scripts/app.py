import paho.mqtt.client as mqtt
import os
import time
from datetime import datetime
import threading
import json
import cv2
from ultralytics import YOLO
import argparse
import glob

# Import local modules
import spot_capture
import yolo_detector
import analog_to_digital
import russ_mqtt_one

# --- Configuration ---
# MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
RESULTS_TOPIC = "spot/results"

# Spot Robot
SPOT_CREDS = {
    "user": "russ",
    "pswd": "12D0ct0r012rmck01",
    "ip": "192.168.2.102",
    "name": "masie"
}
IMAGE_DIR = "new-images"  # Directory to save captured photos
CAPTURE_INTERVAL_SECONDS = 0  # Time between captures

# Processing
YOLO_MODEL_PATH = "yoloe-11l-seg.pt"
yolo_classes_to_detect = ["gauge", "clock"]

ALERT_THRESHOLD = 35.0
WEB_ASSETS_DIR = "web_assets"
PROCESSED_DIR = os.path.join(WEB_ASSETS_DIR, "processed")


def process_image(image_path, yolo_model, mqtt_client):
    """
    Processes a single image in a separate thread.
    Detects the gauge, reads the value, and publishes the results to MQTT.
    """
    print(f"[Thread for {os.path.basename(image_path)}] - Starting processing...")

    # --- Step 1: Detect Gauge ---
    try:
        # Returns: (image with overlay, cropped image, bounding box)
        annotated_img, cropped_img, box = yolo_detector.detect_gauge(image_path, yolo_model)
    except Exception as e:
        print(f"[Thread for {os.path.basename(image_path)}] - Error during gauge detection: {e}")
        return

    # Always save the annotated/original image that is returned from the detector
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(image_path)
    annotated_filename = f"{os.path.splitext(base_name)[0]}_annotated_{timestamp}.jpg"
    annotated_path = os.path.join(PROCESSED_DIR, annotated_filename)
    try:
        if annotated_img is not None:
            cv2.imwrite(annotated_path, annotated_img)
    except Exception as e:
        print(f"[Thread for {os.path.basename(image_path)}] - Error saving annotated image: {e}")

    # Case 1: No gauge was detected
    if cropped_img is None or box is None:
        print(f"[Thread for {os.path.basename(image_path)}] - Gauge not detected.")
        result_data = {
            "status": "Gauge not detected.",
            "value": "N/A",
            "base_image_url": os.path.join("processed", annotated_filename),
            "crop_image_url": None,
            "detection_box": None,
            "timestamp": datetime.now().isoformat()
        }
    # Case 2: Gauge was detected, now process it
    else:
        # --- Step 2: Process Gauge Image (Analog-to-Digital method) ---
        try:
            # value, processed_img = analog_to_digital.convert(cropped_img)
            value, processed_img = russ_mqtt_one.process_image(cropped_img)

        except Exception as e:
            print(f"[Thread for {os.path.basename(image_path)}] - Error during value conversion: {e}")
            value = None
            processed_img = cropped_img  # Use the un-processed crop

        # --- Step 3: Save processed crop image ---
        processed_filename = f"{os.path.splitext(base_name)[0]}_processed_{timestamp}.jpg"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)
        try:
            if processed_img is not None:
                cv2.imwrite(processed_path, processed_img)
        except Exception as e:
            print(f"[Thread for {os.path.basename(image_path)}] - Error saving processed image: {e}")

        # --- Step 4: Determine Status ---
        status = "OK"
        if value is None:
            status = "Could not read value from gauge."
        elif value > ALERT_THRESHOLD:
            status = f"ALERT! Value {value:.2f} > {ALERT_THRESHOLD}"

        result_data = {
            "status": status,
            "value": f"{value:.2f}" if value is not None else "N/A",
            "base_image_url": os.path.join("processed", annotated_filename),
            "crop_image_url": os.path.join("processed", processed_filename),
            "detection_box": box,
            "timestamp": datetime.now().isoformat()
        }

    # --- Final Step: Publish results to MQTT ---
    try:
        mqtt_client.publish(RESULTS_TOPIC, json.dumps(result_data))
        print(f"[Thread for {os.path.basename(image_path)}] - Published results to MQTT.")
    except Exception as e:
        print(f"[Thread for {os.path.basename(image_path)}] - Error publishing to MQTT: {e}")


def main(test_mode=False):
    """
    Main function to start the image capture loop and spawn processing threads.
    """
    print("Starting unified capture and processing service...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- Load Model ---
    try:
        print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.set_classes(yolo_classes_to_detect, yolo_model.get_text_pe(yolo_classes_to_detect))

        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading YOLO model: {e}")
        return

    # --- MQTT Client Setup ---
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()  # Use loop_start for background network handling
        print(f"Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"FATAL: Could not connect to MQTT broker. Error: {e}")
        return

    try:
        if test_mode:
            # --- Test Mode: Loop through local photos ---
            print("Running in TEST MODE.")
            meter_photos_dir = "meter-photos"
            photo_files = sorted(glob.glob(os.path.join(meter_photos_dir, '*.jpg')))
            
            if not photo_files:
                print(f"No photos found in '{meter_photos_dir}'. Exiting test mode.")
                return

            print(f"Found {len(photo_files)} photos to process.")
            
            for image_path in photo_files:
                print("---")
                print(f"[{datetime.now()}] Processing local file: {image_path}")
                
                if os.path.exists(image_path):
                    # Process in a new thread
                    thread = threading.Thread(
                        target=process_image, 
                        args=(image_path, yolo_model, mqtt_client)
                    )
                    thread.start()
                else:
                    print(f"File not found: {image_path}")

                # Wait for 1 second
                time.sleep(1)
            
            print("Finished processing all local photos. Allowing threads time to finish...")
            # Give threads a moment to finish publishing before the script exits
            time.sleep(5) 

        else:
            # --- Live Mode: Capture from Spot ---
            print(f"Starting capture loop. Interval: {CAPTURE_INTERVAL_SECONDS} seconds.")
            while True:
                print("---")
                print(f"[{datetime.now()}] Attempting to capture photo...")

                # --- Step 1: Capture Image ---
                try:
                    original_path = spot_capture.take_photo(SPOT_CREDS, IMAGE_DIR)
                    print(original_path)
                except Exception as e:
                    print(f"Error during spot_capture.take_photo: {e}")
                    original_path = None

                if original_path and os.path.exists(original_path):
                    print(f"Image captured: {original_path}. Spawning processing thread.")
                    # --- Step 2: Process in a new thread ---
                    thread = threading.Thread(
                        target=process_image, 
                        args=(original_path, yolo_model, mqtt_client)
                    )
                    thread.start()
                else:
                    print("Failed to capture image or path is invalid.")

                # --- Wait for next interval ---
                print(f"Waiting for {CAPTURE_INTERVAL_SECONDS} seconds...")
                time.sleep(CAPTURE_INTERVAL_SECONDS)
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        mqtt_client.loop_stop()
        print("MQTT client stopped. Goodbye.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Spot meter reader capture and processing service.")
    parser.add_argument(
        '--test',
        action='store_true',
        help="Run in test mode, processing local files from 'meter-photos' instead of capturing from the robot."
    )
    args = parser.parse_args()
    
    main(test_mode=args.test)