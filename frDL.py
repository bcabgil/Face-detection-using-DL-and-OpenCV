"""This implementation is done by following the online tutorial
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/


Fill test_images_folder variable with the name of the directory where the test images are located

"""


from matplotlib import pyplot as plt
import os
from skimage.transform import resize
from skimage import util
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle, Circle
import cv2


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

# draw the detected faces on the squares
def draw_image_with_boxes(data, result_list):

    #plot the loaded image
    plt.imshow(data)
    #get the context for drawing boxes
    ax = plt.gca()
    #plot each box
    for square in result_list:
        #get coordinates
        x,y,width,height = square["box"]
        #create the shape
        rect = Rectangle((x,y),width, height, fill = False, color="red")
        #draw the box
        ax.add_patch(rect)
        #draw the landmark dots
        for key,value in square["keypoints"].items():
            #create and draw dot
            dot = Circle(value, radius=2, color="red")
            ax.add_patch(dot)
    #show final plot
    plt.show()



# Load all images in test folder
test_images_folder= "test_images"
test_images = load_images_from_folder(test_images_folder)


# Detect the faces from the images
for image in test_images:
    detector = MTCNN()
    # detect the faces in the image
    faces = detector.detect_faces(image)
    # display faces on the original image
    draw_image_with_boxes(image,faces)

