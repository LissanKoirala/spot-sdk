import cv2
import numpy as np
import math

def mask_circular_dial(img):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1], w - center[0], h - center[1]) - 10  # leave a small margin
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img

def mask_needle(gray_img):
    # Threshold to keep only the darkest pixels (needle is usually black)
    _, needle_mask = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY_INV)
    # Morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    needle_mask = cv2.morphologyEx(needle_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return needle_mask

def detect_needle_angle(image_path, debug=False):
    # Load image
    img = cv2.imread(image_path)
    # Remove background by masking the main dial
    img = mask_circular_dial(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Mask to isolate the needle
    needle_mask = mask_needle(blurred)
    # Detect edges only on the needle mask
    edges = cv2.Canny(needle_mask, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None:
        raise ValueError("No lines detected")

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Find the longest line from center outward (assume this is the needle)
    max_len = 0
    best_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist1 = math.hypot(x1 - center[0], y1 - center[1])
        dist2 = math.hypot(x2 - center[0], y2 - center[1])
        if dist1 > max_len or dist2 > max_len:
            max_len = max(dist1, dist2)
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        raise ValueError("Needle not detected")

    x1, y1, x2, y2 = best_line

    # Choose the endpoint farther from center
    if math.hypot(x1 - center[0], y1 - center[1]) > math.hypot(x2 - center[0], y2 - center[1]):
        dx = x1 - center[0]
        dy = center[1] - y1
    else:
        dx = x2 - center[0]
        dy = center[1] - y2

    angle = math.degrees(math.atan2(dy, dx))

    # Convert to 0-360
    angle = (angle + 360) % 360

    if debug:
        print(f"Detected angle: {angle:.2f} degrees")
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img, center, 5, (0, 255, 0), -1)
        cv2.imshow("Needle Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return angle

def angle_to_value(angle, min_angle=45, max_angle=315, min_value=0, max_value=100):
    """
    Map angle from 225° (0°C) to -45° or 315° (120°C).
    Handles wrapping correctly.
    """
    # Normalize angle into range -135° to 135°
    if angle > 225:
        angle -= 360

    ratio = (225 - angle) / 270  # total sweep is 270°
    value = min_value + ratio * (max_value - min_value)
    return round(value, 2)

# Example usage
if __name__ == "__main__":
    image_path = "test.jpg"  # Replace with your image path
    angle = detect_needle_angle(image_path, debug=True)
    value = angle_to_value(angle, min_angle=225, max_angle=-45, min_value=0, max_value=120)
    print(f"Meter Reading: {value}")
