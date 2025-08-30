import cv2
import numpy as np

def detect_gauge(image_path, model):
    """
    Detects the largest gauge in an image, creates a semi-transparent segmentation
    overlay, and returns the annotated image, the cropped gauge, and its coordinates.
    This version geometrically links the correct mask and box.

    Args:
        image_path (str): Path to the input image.
        model: The loaded YOLO model object.

    Returns:
        tuple: A tuple containing:
            - annotated_img (np.ndarray): The image with a segmentation overlay.
            - cropped_img (np.ndarray): The cropped image of the detected gauge.
            - box (list): The [x, y, w, h] of the detection's bounding box.
            Returns (original_image, None, None) if no gauge is detected.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None, None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None, None, None

    # Run YOLO prediction
    results = model(image_path)

    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        print("No gauge detected by YOLO.")
        return image, None, None

    # --- Find the largest mask by area ---
    largest_contour = None
    max_area = 0
    for contour_points in results[0].masks.xy:
        contour = np.array(contour_points, dtype=np.int32)
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if largest_contour is None:
        print("Could not find a valid contour.")
        return image, None, None

    # --- Find the corresponding box by checking which one contains the mask's center ---
    correct_box_coords = None
    
    # Calculate the center of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # Fallback if moments can't be calculated
        cx, cy, _, _ = cv2.boundingRect(largest_contour)
        cx += _ // 2
        cy += _ // 2

    # Find the box that contains this center point
    for yolo_box in results[0].boxes:
        x1, y1, x2, y2 = [int(c) for c in yolo_box.xyxy[0].tolist()]
        if cx > x1 and cx < x2 and cy > y1 and cy < y2:
            correct_box_coords = [x1, y1, x2, y2]
            break
    
    # If no box was found containing the center, fall back to boundingRect as a safety measure
    if correct_box_coords is None:
        print("Could not geometrically link box, falling back to boundingRect.")
        x, y, w, h = cv2.boundingRect(largest_contour)
        correct_box_coords = [x, y, x + w, y + h]

    # --- Use the correct box for UI and cropping ---
    x1, y1, x2, y2 = correct_box_coords
    box_for_ui = [x1, y1, x2 - x1, y2 - y1]
    
    # --- Create annotated image with segmentation overlay ---
    annotated_img = image.copy()
    overlay = image.copy()
    cv2.fillPoly(overlay, [largest_contour], color=(0, 255, 0)) # Green fill
    alpha = 0.4 # Transparency factor
    cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

    # --- Create cropped image using the correct box ---
    cropped_img = image[y1:y2, x1:x2]

    print("Successfully detected and cropped gauge using geometrically linked box.")
    return annotated_img, cropped_img, box_for_ui