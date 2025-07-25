import cv2
import numpy as np
import os
import threading
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Process timed out")

def function_with_timeout(func, args=(), kwargs=None, timeout=10):
    """
    Run any function with a timeout
    
    Args:
        func: Function to execute
        args: Tuple of positional arguments for the function
        kwargs: Dict of keyword arguments for the function
        timeout: Maximum seconds to wait (default: 10)
    
    Returns:
        Function result or None if timeout/error
    """
    if kwargs is None:
        kwargs = {}
        
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Function '{func.__name__}' timed out after {timeout} seconds")
        return None
    
    if exception[0]:
        print(f"Error in function '{func.__name__}': {str(exception[0])}")
        return None
    
    return result[0]

# Your existing functions remain the same...
def avg_circles(circles, b):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        avg_x += circles[0][i][0]
        avg_y += circles[0][i][1]
        avg_r += circles[0][i][2]
    avg_x = int(avg_x / b)
    avg_y = int(avg_y / b)
    avg_r = int(avg_r / b)
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calibrate_gauge(filename, folder_path):
    img = cv2.imread(os.path.join(folder_path, filename + ".jpg"))
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=height // 2,
        param1=100,
        param2=60,
        minRadius=int(height * 0.35),
        maxRadius=int(height * 0.48)
    )

    print("done finding circles")

    if circles is None:
        print("No dial found: could not detect any circular gauge.")
        return None, None, None, None, None, None, None, None

    a, b, c = circles.shape
    x, y, r = avg_circles(circles, b)

    # Draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)

    separation = 10.0
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    for i in range(interval):
        angle_rad = separation * i * np.pi / 180
        p1[i] = [x + 0.9 * r * np.cos(angle_rad), y + 0.9 * r * np.sin(angle_rad)]
        p2[i] = [x + r * np.cos(angle_rad), y + r * np.sin(angle_rad)]
        label_angle_rad = (separation * (i + 9)) * np.pi / 180
        p_text[i] = [x - 10 + 1.2 * r * np.cos(label_angle_rad),
                     y + 5 + 1.2 * r * np.sin(label_angle_rad)]

    for i in range(interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
        cv2.putText(img, str(int(i * separation)), (int(p_text[i][0]), int(p_text[i][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(folder_path, filename + "-calibration.jpg"), img)

    # Hardcoded for demo
    min_angle = 45
    max_angle = 315
    min_value = 0
    max_value = 120
    units = "deg C"

    return min_angle, max_angle, min_value, max_value, units, x, y, r

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, filename, folder_path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dst2 = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)

    print("finding lines")
    lines = cv2.HoughLinesP(dst2, rho=3, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=0)
    print("done finding lines")

    if lines is None:
        print("No lines detected. Check image quality or thresholding.")
        cv2.imwrite(os.path.join(folder_path, filename + "-debug-threshold.jpg"), dst2)
        return None

    # Only consider lines whose endpoints are both within the gauge circle radius
    candidate_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            d1 = dist_2_pts(x, y, x1, y1)
            d2 = dist_2_pts(x, y, x2, y2)
            # Both endpoints must be within the circle (with a small margin)
            if d1 < 0.98*r and d2 < 0.98*r:
                candidate_lines.append([x1, y1, x2, y2, d1, d2])

    if not candidate_lines:
        print("No lines within gauge circle found.")
        return None

    # Pick the longest line among candidates (conservative: must be at least 40% the radius)
    longest_line = None
    max_length = 0
    for x1, y1, x2, y2, d1, d2 in candidate_lines:
        length = dist_2_pts(x1, y1, x2, y2)
        if length > max_length and length > 0.4*r:
            max_length = length
            longest_line = [x1, y1, x2, y2]

    if longest_line is None:
        print("No valid needle found after conservative filtering.")
        return None

    final_line_list = [longest_line]

    x1, y1, x2, y2 = final_line_list[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(folder_path, filename + "-needle.jpg"), img)

    if dist_2_pts(x, y, x1, y1) > dist_2_pts(x, y, x2, y2):
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2

    angle_rad = np.arctan2(y_angle, x_angle)
    angle_deg = np.rad2deg(angle_rad)
    final_angle = (270 - angle_deg) % 360

    old_range = max_angle - min_angle
    new_range = max_value - min_value
    value = ((final_angle - min_angle) * new_range / old_range) + min_value

    return value

def convert(filename):
    try:
        image = cv2.imread(filename)
        if image is None:
            print(f"Could not read image file: {filename}")
            return

        # Enhance contrast
        alpha = 1.5
        beta = 30
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        grey_path = './images/grey.jpg'
        cv2.imwrite(grey_path, adjusted)

        img = cv2.imread(grey_path)
        print('Original Dimensions : ', img.shape)

        # Resize image
        scale_percent = 200
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        resized_path = "./images/resized.jpg"
        cv2.imwrite(resized_path, resized)

        folder_path = "images"
        calib_filename = "resized"

        min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(calib_filename, folder_path)
        if x is None:
            print("Exiting: No gauge detected in the image.")
            return

        img = cv2.imread(os.path.join(folder_path, calib_filename + ".jpg"))
        val = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, calib_filename, folder_path)

        if val is None:
            print("Gauge needle not detected or unreadable.")
            return

        print(f"Current reading: {val:.2f} {units}")

        with open("./images/MyFile.txt", "w") as f:
            f.write(str(val) + "\n")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == '__main__':
    # Use generic timeout wrapper
    function_with_timeout(convert, args=("trial.jpg",), timeout=3)