import cv2
import numpy as np
import os
import time
from datetime import datetime
import shutil
import tkinter as tk
from PIL import Image, ImageTk
import paho.mqtt.client as mqtt

#----------------- CONFIG ----------------
image_name = 'latest.jpg'
archive_dir = 'archive'
output_value_file = 'value.txt'
meter_min_angle = 225  # degrees (0 value)
meter_max_angle = 137  # degrees (120 value)
meter_min_value = 0
meter_max_value = 120
image_size = (500, 500)  # consistent image size
update_interval = 1000  # ms between checks

# MQTT settings
mqtt_broker = "russ-mckay.dyndns.org"
mqtt_port = 1886   # correct port
mqtt_topic = "meter_workorder"
mqtt_payload = "generate_workorder"

# Blur settings
blur_gui = True        # show blurred image in GUI
blur_kernel = 21       # must be odd (e.g., 11, 15, 21, 31)
blur_archive = False   # True = archive blurred version, False = archive original
# -----------------------------------------

# Ensure archive directory exists
os.makedirs(archive_dir, exist_ok=True)

# MQTT client setup
client = mqtt.Client()
try:
    client.connect(mqtt_broker, mqtt_port, 60)
    mqtt_connected = True
except Exception as e:
    print(f"MQTT connection failed: {e}")
    mqtt_connected = False

# Track if work order has already been sent
workorder_sent = False


def process_image():
    """Process latest.jpg if it exists, return value and Tkinter image."""
    if not os.path.exists(image_name):
        return None, None

    img = cv2.imread(image_name)
    if img is None:
        return None, None

    # Resize
    img = cv2.resize(img, image_size)

    # Preprocess for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=80, maxLineGap=20)

    val = -1
    if lines is not None:
        # Pick longest line
        longest_line = max(lines, key=lambda line: np.linalg.norm(
            [line[0][0]-line[0][2], line[0][1]-line[0][3]]))
        x1, y1, x2, y2 = longest_line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Angle calculation
        center = (img.shape[1]//2, img.shape[0]//2)
        d1 = np.linalg.norm([x1-center[0], y1-center[1]])
        d2 = np.linalg.norm([x2-center[0], y2-center[1]])
        needle_end = (x1, y1) if d1 > d2 else (x2, y2)

        needle_vector = np.array(
            [needle_end[0] - center[0], center[1] - needle_end[1]])  # y inverted
        angle_rad = np.arctan2(needle_vector[1], needle_vector[0])
        angle_deg = np.degrees(angle_rad)

        angle_deg = (90 - angle_deg) % 360

        # Map angle to value
        arc_span = 360 - meter_min_angle + meter_max_angle
        if angle_deg >= meter_min_angle:
            val = (angle_deg - meter_min_angle) * \
                (meter_max_value - meter_min_value) / arc_span
        else:
            val = (angle_deg + (360 - meter_min_angle)) * \
                (meter_max_value - meter_min_value) / arc_span

        val = np.clip(val, meter_min_value, meter_max_value)
        val = round(val, 2)

    # Save value
    with open(output_value_file, "w") as f:
        f.write(str(val))

    # Archive image with value + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_filename = f"latest_{val}_{timestamp}.jpg"
    archive_path = os.path.join(archive_dir, archive_filename)

    try:
        if blur_archive:
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            img_for_archive = cv2.GaussianBlur(img, (k, k), 0)
            cv2.imwrite(archive_path, img_for_archive)
            os.remove(image_name)
        else:
            shutil.move(image_name, archive_path)
    except Exception as e:
        print(f"Archiving failed: {e}")

    # Convert image for display (with optional blur)
    display_img = img.copy()
    if blur_gui:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        display_img = cv2.GaussianBlur(display_img, (k, k), 0)

    img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tk_img = ImageTk.PhotoImage(pil_img)

    return val, tk_img


def update_gui():
    """Periodic updater for GUI."""
    global workorder_sent

    val, tk_img = process_image()
    if val is not None:
        value_label.config(text=f"Meter Value: {val}")
        image_label.config(image=tk_img)
        image_label.image = tk_img

        if val <= 25:
            notify_label.config(
                text="Temperature OK ✅",
                fg="green"
            )
            workorder_sent = False  # reset flag when safe again
            clear_button.config(state="disabled")

        else:
            # Always show warning
            msg = "Warning! Value exceeds safe working temperature ⚠️"
            color = "red"

            # Send MQTT work order only once per threshold crossing
            if mqtt_connected and not workorder_sent:
                try:
                    client.publish(mqtt_topic, mqtt_payload)
                    msg += "\nInspection work order generated in Maximo"
                    color = "blue"
                    workorder_sent = True
                except Exception as e:
                    msg += f"\nMQTT publish failed: {e}"
                    color = "orange"

            notify_label.config(
                text=msg,
                fg=color,
                justify="center"
            )
            clear_button.config(state="normal")

    root.after(update_interval, update_gui)


def clear_alert():
    """Clear alert and reset MQTT work order flag."""
    global workorder_sent
    workorder_sent = False
    notify_label.config(
        text="Alerts cleared. Monitoring resumed...",
        fg="black"
    )
    clear_button.config(state="disabled")


def on_exit():
    """Handle exit: close Tkinter + OpenCV windows."""
    cv2.destroyAllWindows()
    root.destroy()


# --------------- GUI Setup ---------------
root = tk.Tk()
root.title("Analogue Meter Reader")
root.geometry("600x900")

title_label = tk.Label(root, text="Analogue Meter Reader",
                       font=("Arial", 20, "bold"))
title_label.pack(pady=10)

value_label = tk.Label(root, text="Waiting for image...",
                       font=("Arial", 18))
value_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

notify_label = tk.Label(root, text="System idle...",
                        font=("Arial", 16),
                        justify="center")
notify_label.pack(pady=15)

# Clear alert button (disabled until alert appears)
clear_button = tk.Button(
    root,
    text="Clear Alert",
    font=("Arial", 14, "bold"),
    bg="lightgray",
    fg="black",
    state="disabled",
    command=clear_alert
)
clear_button.pack(pady=10)

# Exit button full width at bottom
exit_button = tk.Button(
    root,
    text="Exit",
    font=("Arial", 16, "bold"),
    bg="white",   # white background
    fg="black",   # black text
    command=on_exit
)
exit_button.pack(side="bottom", fill="x", padx=20, pady=20)

# Start update loop
root.after(1000, update_gui)
root.mainloop()
