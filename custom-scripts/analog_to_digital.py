import cv2
import numpy as np

def _avg_circles(circles):
    """Helper to average circle properties."""
    avg_x = np.mean(circles[0, :, 0])
    avg_y = np.mean(circles[0, :, 1])
    avg_r = np.mean(circles[0, :, 2])
    return int(avg_x), int(avg_y), int(avg_r)

def _dist_2_pts(x1, y1, x2, y2):
    """Helper to calculate distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def _calibrate_gauge(image):
    """
    Finds the gauge dial in the image.
    """
    height, _ = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=height // 2,
        param1=100, param2=60, minRadius=int(height * 0.35), maxRadius=int(height * 0.48)
    )

    if circles is None:
        return None, None, None, None, None, None, None, None

    x, y, r = _avg_circles(circles)

    min_angle = 45
    max_angle = 315
    min_value = 0
    max_value = 120
    units = "deg C"

    return min_angle, max_angle, min_value, max_value, units, x, y, r

def _get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r):
    """Finds the needle and calculates the gauge value."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dst2 = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)

    lines = cv2.HoughLinesP(dst2, rho=3, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=0)

    if lines is None:
        return None, img

    candidate_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if _dist_2_pts(x, y, x1, y1) < r and _dist_2_pts(x, y, x2, y2) < r:
            candidate_lines.append(line[0])

    if not candidate_lines:
        return None, img

    longest_line = max(candidate_lines, key=lambda line: _dist_2_pts(line[0], line[1], line[2], line[3]))

    if _dist_2_pts(longest_line[0], longest_line[1], longest_line[2], longest_line[3]) < 0.4 * r:
        return None, img

    x1, y1, x2, y2 = longest_line
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    p1_dist = _dist_2_pts(x, y, x1, y1)
    p2_dist = _dist_2_pts(x, y, x2, y2)
    
    x_angle, y_angle = (x1 - x, y - y1) if p1_dist > p2_dist else (x2 - x, y - y2)

    angle_rad = np.arctan2(y_angle, x_angle)
    angle_deg = np.rad2deg(angle_rad)
    final_angle = (270 - angle_deg) % 360

    old_range = max_angle - min_angle
    new_range = max_value - min_value
    value = (((final_angle - min_angle) * new_range) / old_range) + min_value

    return value, img

def convert(image: np.ndarray):
    """
    Processes an image of a gauge to read its value.

    Args:
        image: A numpy array representing the image (in BGR format).

    Returns:
        A tuple containing:
        - The calculated value (float) or None if reading fails.
        - The processed image (numpy array) with visualizations.
    """
    if image is None:
        return None, None

    try:
        adjusted = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        
        width = int(adjusted.shape[1] * 2)
        height = int(adjusted.shape[0] * 2)
        resized = cv2.resize(adjusted, (width, height), interpolation=cv2.INTER_AREA)

        calib_result = _calibrate_gauge(resized)
        min_angle, max_angle, min_value, max_value, units, x, y, r = calib_result

        if x is None:
            return None, resized

        cv2.circle(resized, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(resized, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)

        val, final_image = _get_current_value(resized, min_angle, max_angle, min_value, max_value, x, y, r)

        if val is not None:
            print(f"Current reading: {val:.2f} {units}")
        
        return val, final_image

    except Exception as e:
        print(f"Error during conversion: {e}")
        return None, locals().get('resized', image)
