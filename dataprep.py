from imageai.Detection import ObjectDetection
import os
from PIL import Image
import numpy as np
import cv2


# directory that contains all images
inputdir = '/Users/PGGAN/thecarconnectionpicturedataset/'
# output directory
outputdir = '/Users/PGGAN/processeddata/'


# create detector object using imageai
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
#specify object detection model to use and path
detector.setModelPath('/Users/PGGAN/resnet50_coco_best_v2.1.0.h5')
detector.loadModel(detection_speed="fast")
custom_objects = detector.CustomObjects(car=True, truck=True)


def load_image(filename):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    # convert to array
    pixels = np.asarray(image)
    return pixels


def extract_object(pixels):
    detected_image_array, detections = detector.detectObjectsFromImage(input_type="array", output_type="array",
                                                                       custom_objects=custom_objects,
                                                                       input_image=pixels,
                                                                       display_percentage_probability=False,
                                                                       display_object_name=False, display_box=False)
    if not detections:
        return None
    details = detections[0].get("box_points")
    x1, y1, x2, y2, = details[0], details[1], details[2], details[3]
    carpixels = pixels[y1:y2, x1:x2]
    return carpixels

#resizes image and adds padding accordingly
def add_padding(im):
    desired_size = 256
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def load_cars(directory, outputdir):
    cars = list()
    # enumerate files
    for filename in os.listdir(directory):
        # load the image
        pixels = load_image(directory + filename)
        # get car
        car = extract_object(pixels)
        if car is None:
            continue
        car = add_padding(car)
        data = Image.fromarray(car)
        #make output directory and save each image
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        data.save(outputdir + filename)
        cars.append(car)
        # store
        print(len(cars), car.shape)


# load and extract all faces
load_cars(inputdir, outputdir)
