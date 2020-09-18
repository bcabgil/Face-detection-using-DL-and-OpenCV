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
def draw_image_with_boxes(data, result_list,i):

    # We add cv2.cvtColor to change the images from BGR as loaded with cv2 to RGB to be read by plt
    plt.imshow( cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
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
    plt.axis("off")
    plt.margins(0, 0)
    plt.savefig("result{}".format(i),bbox_inches='tight')
    plt.show()

#draw found faces subplot
def draw_faces(data, result_list,j):
    #iterate through the found faces
    for i in range(len(result_list)):
        #get coordinates
        x1,y1, width, height = result_list[i]["box"]
        x2, y2 = x1 + width, y1 + height
        #define subplot
        plt.subplot(1, len(result_list), i+1)
        plt.axis("off")
        plt.margins(0,0)
        #plot face
        plt.imshow(cv2.cvtColor(data[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))
    plt.savefig("faces{}.jpg".format(j),bbox_inches='tight')
    plt.show()
    return


# Load all images in test folder
working_dir = os.path.abspath(os.getcwd())

general_dir = working_dir.rsplit("\\",1)[0]

test_images_folder= os.path.join(general_dir,"test_images")

test_images = load_images_from_folder(test_images_folder)


# Detect the faces from the images
for i,image in enumerate(test_images):
    detector = MTCNN()
    # detect the faces in the image
    faces = detector.detect_faces(image)
    # display faces on the original image
    draw_image_with_boxes(image,faces,i)
    if faces:
        draw_faces(image,faces,i)

