import cv2
import numpy as np
import math

def detect_needle_angle(image_path, debug=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect circular dial
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=100, maxRadius=200)
    if circles is None:
        raise ValueError("No circular dial detected")

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    center = (x, y)

    # Detect lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

    if lines is None:
        raise ValueError("No lines detected")

    needle = None
    max_len = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        dist = min(math.hypot(x1 - x, y1 - y), math.hypot(x2 - x, y2 - y))
        if length > max_len and dist < r * 0.2:
            max_len = length
            needle = (x1, y1, x2, y2)

    if needle is None:
        raise ValueError("Needle not detected")

    x1, y1, x2, y2 = needle
    if math.hypot(x2 - x, y2 - y) > math.hypot(x1 - x, y1 - y):
        dx = x2 - x
        dy = y - y2
    else:
        dx = x1 - x
        dy = y - y1

    angle = math.degrees(math.atan2(dy, dx))
    angle = (angle + 360) % 360

    if debug:
        img_copy = img.copy()
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img_copy, center, 5, (0, 255, 0), -1)
        cv2.imshow("Debug", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return angle

def angle_to_value(angle, min_angle=225, max_angle=-45, min_value=0, max_value=120):
    if angle > 225:
        angle -= 360
    ratio = (225 - angle) / 270
    value = min_value + ratio * (max_value - min_value)
    return round(value, 1)

if __name__ == "__main__":
    image_path = "test.jpg"
    angle = detect_needle_angle(image_path, debug=True)
    value = angle_to_value(angle)
    print(f"Meter Reading: {value} °C")
