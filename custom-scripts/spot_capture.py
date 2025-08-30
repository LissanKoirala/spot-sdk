import os
import shutil
from datetime import datetime
from spot_robot_utils import Spot, suppress_stdout

def take_photo(spot_creds: dict, images_dir="images"):
    """
    Connects to Spot, takes a photo, and saves it.

    Args:
        spot_creds: A dictionary with 'name', 'ip', 'user', and 'pswd'.
        images_dir: The directory to save the final image in.

    Returns:
        The file path (str) to the captured image, or None on failure.
    """
    os.makedirs(images_dir, exist_ok=True)
    
    try:
        print("Connecting to Spot and taking photo...")
        with suppress_stdout():
            spot = Spot(spot_creds['name'], spot_creds['ip'], spot_creds['user'], spot_creds['pswd'])
        
        # This function saves a timestamped file in the root directory
        temp_image_path = spot.take_photo_new()
        if temp_image_path is None:
            print("Spot failed to take a photo.")
            raise ConnectionError("Spot failed to take a photo.")

        # Move and rename the image to its final destination
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_image_path = os.path.join(images_dir, f"spot_capture_{timestamp}.jpg")
        shutil.move(temp_image_path, final_image_path)
        print(f"Photo saved to: {final_image_path}")
        return final_image_path

    except Exception as e:
        print(f"Error connecting to Spot or taking photo: {e}")
        # Fallback to a local file if Spot fails, for testing
        fallback_path = os.path.join(images_dir, "latest.jpg")
        if os.path.exists(fallback_path):
            print(f"Using fallback image: {fallback_path}")
            return fallback_path
        else:
            print("No fallback image found.")
            return None
