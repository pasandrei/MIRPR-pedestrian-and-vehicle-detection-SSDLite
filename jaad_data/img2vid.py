import cv2
import argparse
import os

ext = 'png'
rootdir = '.'

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

for dir_path, _, files in os.walk(rootdir):
    try:
        dir_path.index("video") 
        #dir_path.index("inference")
    except Exception as e:
        print(dir_path + " does not contain images with bboxes")
        continue
    output = dir_path + '.mp4'
    if os.path.isfile(output):
        continue
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    images.sort(key=lambda name: int(name.split(".")[0]))

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 30.0, (width, height))

    for image in images:
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))