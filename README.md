# Face-detection-using-DL-and-OpenCV
This repository contains some work to increase my knowledge in the field of face recognition and face detection.

Content of the repository:

* Test images: Contain the images used to test the models
* Face Detection using OpenCV: 
    - The Method used in this code is the OpenCV2 implementation of CascadeClassifier.
    - Contains a Jupyter notebook to understand and follow the code easyer and in a more interactive way.
    - Contains a .py file with the same code but structured to be used as part of a greater project.
* Face Detection using Deep Learning:
    - Uses an adaptation from the Facenet's MTCNN implementation. The implementation and model is obtained from https://github.com/jbrownlee/mtcnn.
    - Contains a Jupyter notebook to understand and follow the code easyer and in a more interactive way.
    - Contains a .py file with the same code but structured to be used as part of a greater project.
    

## Lessons Learned
* CascadeClassifier from OpenCV can be adjusted to each context, however, it is less sensitive when detecting faces.
* FaceNet, can also be trained to learn how to accurately predict a certain kind of image. However, when taking the pretrained model it has better accuracy than the CascadeClassifier.

