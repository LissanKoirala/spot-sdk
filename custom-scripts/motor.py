from spot_robot_utils import *
import sys
import os
import shutil
import glob
import analog_to_digital
import time  # Optional: if you want a delay between loops

########### CREDENTIALS SECTION #####################  
user = "russ"
pswd = "12D0ct0r012rmck01"
spot_ip = "192.168.80.3"
spot_name = "masie"

# Ensure images directory exists
os.makedirs("images", exist_ok=True)

while True:
    print("Starting the spot.take_photo process")

    # Create Spot object and take photo
    with suppress_stdout():
        spot = Spot(spot_name, spot_ip, user, pswd)

    spot.take_photo_new()
    print("Photo taken")

    # Remove old 'latest.jpg' if it exists
    latest_img_path = "images/latest.jpg"
    if os.path.exists(latest_img_path):
        os.remove(latest_img_path)

    # Move the new JPG (assumes only one new JPG created per cycle)
    new_images = [f for f in os.listdir() if f.lower().endswith('.jpg') and not f.startswith("images/")]

    if not new_images:
        print("No new image found!")
        continue

    # Move and rename the image
    new_image = new_images[0]
    new_path = os.path.join("images", new_image)
    shutil.move(new_image, new_path)

    os.rename(new_path, latest_img_path)
    print(f"Moved and renamed {new_image} to {latest_img_path}")

    # Convert image to digital reading
    try:
        # value = analog_to_digital.convert(latest_img_path)
        value = analog_to_digital.function_with_timeout(analog_to_digital.convert, args=(latest_img_path,), timeout=3)
        print(f"Meter reading: {value}")
    except Exception as e:
        print(f"Error converting image to reading: {e}")

    # Add a short delay between iterations
    # time.sleep(5)
