import cv2
import numpy as np

def process_image(image: np.ndarray):
    """
    Processes an image of a gauge to read its value using the 'russ_mqtt_one' style.

    Args:
        image: A numpy array representing the image (in BGR format).

    Returns:
        A tuple containing:
        - The calculated value (float) or None if reading fails.
        - The processed image (numpy array) with visualizations.
    """
    if image is None:
        return None, None

    # Configuration
    meter_min_angle = 225
    meter_max_angle = 137
    meter_min_value = 0
    meter_max_value = 120
    image_size = (500, 500)

    try:
        img = cv2.resize(image, image_size)

        # Preprocess for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=80, maxLineGap=20)

        val = None
        if lines is not None:
            longest_line = max(lines, key=lambda line: np.linalg.norm(
                [line[0][0] - line[0][2], line[0][1] - line[0][3]]))
            x1, y1, x2, y2 = longest_line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Angle calculation
            center = (img.shape[1] // 2, img.shape[0] // 2)
            d1 = np.linalg.norm([x1 - center[0], y1 - center[1]])
            d2 = np.linalg.norm([x2 - center[0], y2 - center[1]])
            needle_end = (x1, y1) if d1 > d2 else (x2, y2)

            needle_vector = np.array([needle_end[0] - center[0], center[1] - needle_end[1]])
            angle_rad = np.arctan2(needle_vector[1], needle_vector[0])
            angle_deg = (90 - np.degrees(angle_rad)) % 360

            # Map angle to value
            arc_span = 360 - meter_min_angle + meter_max_angle
            if angle_deg >= meter_min_angle:
                val = (angle_deg - meter_min_angle) * (meter_max_value - meter_min_value) / arc_span
            else:
                val = (angle_deg + (360 - meter_min_angle)) * (meter_max_value - meter_min_value) / arc_span
            
            val = np.clip(val, meter_min_value, meter_max_value)
            val = round(val, 2)

        return val, img

    except Exception as e:
        print(f"Error in russ_mqtt_style processing: {e}")
        return None, image
