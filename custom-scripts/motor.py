from spot_robot_utils import *
import sys
import os
import shutil
import glob
import analog_to_digital

while True:

########### CREDENTIALS SECTION #####################  
    user = "russ"
    pswd = "12D0ct0r012rmck01"

#   Autonomous mode - Spot as Access Point
#   spot_ip = "192.168.80.3"

#   York Road Spot_london Access Point
    spot_ip = "192.168.2.102"

    spot_name = "masie"

    print ("Starting the spot.take_photo process")
    # Create Robot Object
    with suppress_stdout():
      spot = Spot(spot_name, spot_ip, user, pswd)
    spot.take_photo_new()
    print ("Taking Photo")
# Rename latest jpg to gauge.jpg
# First, clean the images directory - remove latest.jpg, but check that it exists first

    if os.path.exists("images/latest.jpg"):
        os.remove("images/latest.jpg")
    else:
        print("The file does not exist")

    # Next, look for a jpg image - there should be one creted using the spot.take_photo_new() function - it's a random name!        
        images = [f for f in os.listdir() if '.jpg' in f.lower()]

    # Next, move the image into the images directory
        for image in images:
            print (image)
            new_path = 'images/' + image
            shutil.move(image, new_path)

    # Now change the file name of the latest (should be the only) file to latest.jpg
            list_of_files = glob.glob('images/*.jpg') 
            latest_file = max(list_of_files, key=os.path.getctime)
            print(latest_file)
            os.rename(latest_file, 'images/latest.jpg')

            value = analog_to_digital.convert("images/latest.jpg")
            print(f"Meter reading {value}")