"""This implementation is done by following the online tutorial
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/

Please download the haarcascade frontalface detection file from Face Detection using OpenCV GitHub ->
https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
 and save it in the working directory as ‘haarcascade_frontalface_default.xml‘.

Fill test_images_folder variable with the name of the directory where the test images are located

"""

import cv2
from matplotlib import pyplot as plt
import os
from skimage.transform import resize
from skimage import util
import numpy as np
working_dir = os.path.abspath(os.getcwd())

general_dir = working_dir.rsplit("\\",1)[0]

test_images_folder= os.path.join(general_dir,"test_images")

def load_images_from_folder(test_images_folder):
    images=[]

    for filename in os.listdir(test_images_folder):
        img=cv2.imread(os.path.join(test_images_folder,filename))

        if img.shape[0]> 1000 or img.shape[1]>1000:
            img = resize(img, (img.shape[0] // 4, img.shape[1] // 4,3),
                       anti_aliasing=True)

        # CascadeClassifier needs the image to be uint8 and skimage.resize transformes it into float64
        img = util.img_as_ubyte(img)

        if img is not None:
            images.append(img)
    return images

def face_detection(image):
    # Load the pretrained model form Face Detection using OpenCV
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # perform the face detection
    bboxes = classifier.detectMultiScale(image)

    # print the bounding box for each detected face
    for box in bboxes:
        # extract and calculate the boxes parameters
        x, y, width, height = box
        x2, y2 = x + width, y + height

        # draw a rectangle over the pixels
        cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 3)

    return image


# Load all images in test folder
test_images = load_images_from_folder(test_images_folder)


# Detect the faces from the images
for i,image in enumerate(test_images):
    image_detected = face_detection(image)
    cv2.imwrite("image{}.jpg".format(i), image_detected)
    cv2.imshow("Face detected",image_detected)

    # keep the window open until we press a key
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()
