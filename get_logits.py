#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 01:29:32 2024

@author: behnam
"""



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.keras.models import Model



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:15:34 2020

@author: behnam
"""



# from keras.utils import multi_gpu_model

import numpy as np
import sys
sys.setrecursionlimit(1000)

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
from tensorflow.python.client import device_lib
device_lib.list_local_devices()# Check if we're using a GPU device

# import efficientnet.tfkeras

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNet , MobileNetV2, Xception
import tensorflow as tf
from tensorflow.keras.models import clone_model
model = load_model('imagenet_16_120_transfer_server_model.h5')
# model.load_weights( 'imagenet_16_120_inception_resnet_v2_weights_epoch_40.h5' )
print ('model loaded successfully')

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
from image_preprocessing_ver1 import ImageDataGenerator as ImageDataGenerator_1
# remove softmax
# model.layers.pop()


# now model outputs logits

last_layer = model.layers[-1]

# Remove the activation function from the last layer
last_layer.activation = None

##########################################    
from PIL import Image
from utils import get_datasets
train_data, test_data = get_datasets('/home/behnam/Desktop/Implementation_imagenet/ImageNet16')


### test image to see if it is correct
img = train_data.data[0]
img = Image.fromarray(img)
img.show()

# 
x_train = np.array(train_data.data)


img = x_train[0]
img = Image.fromarray(img)
img.show()


y_train = np.array(train_data.targets)

x_test = np.array(test_data.data)
y_test = np.array(test_data.targets)




# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_val   = x_train[-6000:,:,:,:]
y_val   = y_train[-6000:]

x_train = x_train[:-6000,:,:,:]
y_train = y_train[:-6000]



def preprocess_data(X, Y):
    """
    function that pre-processes the CIFAR10 dataset as per
    densenet model requirements for input images
    labels are one-hot encoded
    """
    X = tf.keras.applications.inception_resnet_v2.preprocess_input(X)
    # Y = tf.keras.utils.to_categorical(Y - 1, num_classes=120)
    # Y = tf.keras.utils.to_categorical(Y)
    return X

x_train = preprocess_data(x_train, y_train)
x_test = preprocess_data(x_test, y_test)

train_datagen = ImageDataGenerator_1(
                                          # rotation_range=40,
                                          # width_shift_range=0.2,
                                          # height_shift_range=0.2,
                                          # shear_range=0.2,
                                          # zoom_range=0.2,
                                          rescale=1./255.,
                                          horizontal_flip=True)
                                          # ,fill_mode='nearest')
                                          
train_generator = train_datagen.flow(x_train,y_train,batch_size=32, shuffle=False)


val_datagen = ImageDataGenerator_1(horizontal_flip=True)
val_generator = val_datagen.flow(x_test,
                                 y_test,
                                 batch_size=32 , shuffle=False)


batches = 0
val_logits = {}
index = 0

for x_batch, name_batch in tqdm(val_generator):
    
    batch_logits = model.predict_on_batch(x_batch)
    
    for i, n in enumerate(name_batch):
        key = 'image_%d_class_%d' % (index,n)
        val_logits[index] = batch_logits[i]
        
        index += 1
    batches += 1
    if batches >= 187.5: # 6000/32
        break
# val_logits = np.array(val_logits)


batches = 0
train_logits = {}
index = 0
for x_batch, name_batch in tqdm(train_generator):
    
    batch_logits = model.predict_on_batch(x_batch)
    
    for i, n in enumerate(name_batch):
        key = 'image_%d_class_%d' % (index,n)
        train_logits[index] = batch_logits[i]
        index += 1
    batches += 1
    if batches >= 4553.125: # 151700/32
        break
# train_logits = np.array(train_logits)







np.save( 'train_logits_resnet_inception_imagenet_16_120.npy', train_logits)
np.save( 'val_logits_resnet_inception_imagenet_16_120.npy', val_logits)

