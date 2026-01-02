# import standard dependencies
import cv2 # computer vision
import numpy as np # arrays and matrices
import os # operating system
import random # randomization
import tensorflow as tf # tensorflow, deep learning framework
from matplotlib import pyplot as plt # plotting library
# import tensorflow dependencies - functional api
from tensorflow import keras # functional api
from keras.models import Model 
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
# Import uuid library to generate unique image names
import uuid

# stop tensorflow from allocating all gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: # accessing all of the gpus on the physical machine
   tf.config.experimental.set_memory_growth(gpu, True)

# setup paths
positivePath = os.path.join('data', 'positive')
negativePath = os.path.join('data', 'negative')
anchorPath = os.path.join('data', 'anchor')

# make the directories
os.makedirs(positivePath)
os.makedirs(negativePath)
os.makedirs(anchorPath)

# move images to the positive and negative folders
for directory in os.listdir('lfw-deepfunneled'):
    for file in os.listdir(os.path.join('lfw-deepfunneled', directory)):
       exPath = os.path.join('lfw-deepfunneled', directory, file)
       newPath = os.path.join(negativePath, file)
       os.replace(exPath, newPath)

# setup video capture device
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read() # read the frame from the camera
 
    frame = frame[120:120+250, 200:200+250, :] # crop the frame to a 250x250 square

    if cv2.waitKey(1) & 0xFF == ord('a'): # if 'a' is pressed, save the anchor image
        # create the unique file path
        imgname = os.path.join(anchorPath, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame) # write the image to file

    if cv2.waitKey(1) & 0xFF == ord('p'): # if 'p' is pressed, save the positive image
        # create the unique file path
        imgname = os.path.join(positivePath, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame) # write the image to file

    
    cv2.imshow('Image Collection', frame) # display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # quit the loop if 'q' is pressed
        break

cap.release()
cap.destroyAllWindows()

anchor = tf.data.Dataset.list_files(anchorPath+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(positivePath+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(negativePath+'\*.jpg').take(300)

dir_test = anchor.as_numpy_iterator()
print(dir_test.next())

def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0
    
    # Return image
    return img


img = preprocess('data\\anchor\\a4e73462-135f-11ec-9e6e-a0cec8d2d278.jpg')
img.numpy().max()
plt.imshow(img)
dataset.map(preprocess)




