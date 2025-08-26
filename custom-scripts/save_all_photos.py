from spot_robot_utils import *
import os
import shutil
import time

# Credentials and robot configuration (copied from motor.py)
user = "russ"
pswd = "12D0ct0r012rmck01"
spot_ip = "192.168.2.102"
spot_name = "masie"

# Ensure destination directory exists
DEST_DIR = "meter-photos"
os.makedirs(DEST_DIR, exist_ok=True)


def get_next_index(destination_directory: str) -> int:
    """Return the next sequential index based on existing JPG files in the destination directory.

    This scans for files ending in .jpg or .jpeg and uses a simple count to continue the
    sequence; gaps are ignored.
    """
    existing = [
        f
        for f in os.listdir(destination_directory)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]
    return len(existing)


def save_new_jpegs_sequentially(source_directory: str, destination_directory: str, start_index: int) -> int:
    """Move any JPGs from source_directory into destination_directory with sequential names.

    Returns the next index after processing.
    """
    index = start_index
    new_images = [
        f
        for f in os.listdir(source_directory)
        if f.lower().endswith((".jpg", ".jpeg")) and not os.path.isdir(f)
    ]

    # Sort for stable ordering (by filename)
    new_images.sort()

    for filename in new_images:
        src_path = os.path.join(source_directory, filename)

        # Skip images that already live in destination directory
        if os.path.dirname(src_path) == os.path.abspath(destination_directory):
            continue

        dest_filename = f"photo_{index:06d}.jpg"
        dest_path = os.path.join(destination_directory, dest_filename)

        # Ensure we don't overwrite; advance index until available
        while os.path.exists(dest_path):
            index += 1
            dest_filename = f"photo_{index:06d}.jpg"
            dest_path = os.path.join(destination_directory, dest_filename)

        shutil.move(src_path, dest_path)
        print(f"Saved {filename} -> {dest_path}")
        index += 1

    return index


def main() -> None:
    # Establish starting index based on what's already in DEST_DIR
    next_index = get_next_index(DEST_DIR)

    print("Starting sequential photo saver to 'meter-photos'. Press Ctrl+C to stop.")

    while True:
        print("Capturing photo from Spot...")

        # Create Spot object and capture a photo
        with suppress_stdout():
            spot = Spot(spot_name, spot_ip, user, pswd)

        spot.take_photo_new()
        print("Photo captured.")

        # Move any new JPGs in the current working directory into DEST_DIR sequentially
        next_index = save_new_jpegs_sequentially(os.getcwd(), DEST_DIR, next_index)

        # Optional delay between captures; adjust as needed
        # time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")

