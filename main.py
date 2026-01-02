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


