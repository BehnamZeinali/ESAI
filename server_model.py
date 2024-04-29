#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 22:41:49 2021

@author: behnam
"""

import tensorflow as tf
tf.__version__

import numpy as np
import matplotlib.pyplot as plt
# import cv2 as cv

np.set_printoptions(precision=7)


# import tensorflow_datasets as tfds

from tensorflow.keras import Model
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from tensorflow.python.keras.models import Model, Sequential, model_from_config, load_model
from tensorflow.keras.applications import InceptionResNetV2

model = load_model('imagenet_16_120_transfer_inception_resnet_v2.h5')

from utils import get_datasets
train_data, test_data = get_datasets('./ImageNet16')


x_train = np.array(train_data.data)
y_train = np.array(train_data.targets)

x_test = np.array(test_data.data)
y_test = np.array(test_data.targets)


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
    Y = tf.keras.utils.to_categorical(Y - 1, num_classes=120)
    # Y = tf.keras.utils.to_categorical(Y)
    return X, Y

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# X_val, y_val = preprocess_data(X_val, y_val)


base_inception = InceptionResNetV2(include_top=False,
                             input_shape=(224, 224, 3))

# for layer in base_inception.layers:
#     layer.trainable = False

input_layer = Input(shape=(16, 16, 3))

resizing_layer = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_layer)


# resizing_layer = Lambda(lambda image:                     \
#                 K.preprocessing.image.smart_resize(image, \
#                 (224, 224)))(input_layer)

inception_layers = base_inception(resizing_layer, training=False)

glob_pooling = GlobalAveragePooling2D()(inception_layers)
layer_i = Dense(500, activation='relu')(glob_pooling)
dropout_layer = Dropout(0.3)(layer_i)
output_layer =  Dense(120, activation='softmax')(dropout_layer)

model = Model(inputs=input_layer, outputs=output_layer)


model.summary()






model.compile(
         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
         loss='categorical_crossentropy',
         metrics=['accuracy'])


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                          factor=0.6,
                                          patience=2,
                                          verbose=1,
                                          mode='max',
                                          min_lr=1e-7)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                       patience=10,
                                       verbose=1,
                                       mode='max')
checkpoint = tf.keras.callbacks.ModelCheckpoint('inception_imagenet16_120.keras',
                                         monitor='val_accuracy',
                                         verbose=1,
                                         save_weights_only=False,
                                         save_best_only=True,
                                         mode='max',
                                         save_freq='epoch')

checkpoint = tf.keras.callbacks.ModelCheckpoint('imagenet_16_120_transfer_server_model.h5',
                                         monitor='val_accuracy',
                                         verbose=1,
                                         save_weights_only=False,
                                         save_best_only=True,
                                         mode='max',
                                         save_freq='epoch')
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                          # rotation_range=40,
                                          # width_shift_range=0.2,
                                          # height_shift_range=0.2,
                                          # shear_range=0.2,
                                          # zoom_range=0.2,
                                          horizontal_flip=True)
                                          # ,fill_mode='nearest')
                                          
train_generator = train_datagen.flow(x_train,y_train,batch_size=32)


val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
val_generator = val_datagen.flow(x_test,
                                 y_test,
                                 shuffle=False,
                                 batch_size=32)
train_steps_per_epoch = x_train.shape[0] // 32
val_steps_per_epoch = x_test.shape[0] // 32
history = model.fit(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    epochs=50,
                    shuffle=True,
                    callbacks=[lr_reduce, early_stop, checkpoint],
                    verbose=1)  


plt.figure(1, figsize = (15,8))
plt.subplot(221)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.subplot(222)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.show()

model.save('last_model_imagenet.h5')

model.save_weights('server_model_saved_weights.h5')




batch_size=32
server_predict = []
true_label = []

test_generator = val_generator
for i in range(len(test_generator)):
    inputs_batch, labels_batch = test_generator[i]  # Retrieve batch of images and labels
    x_train = model.predict(inputs_batch)
    true_label.append(labels_batch)
    server_predict.append( x_train)
    
server_predict = np.concatenate(server_predict, axis=0)
true_label = np.concatenate(true_label, axis=0)


   
classified = np.zeros((len(true_label),1));
label = np.zeros((len(true_label),1));
for i in range(0, len(classified)):
        classified[i] = np.where(server_predict[i] == np.amax(server_predict[i])) 
        label[i] = (np.where(true_label[i] == np.amax(true_label[i])))    
        
        
from sklearn.metrics import accuracy_score
scores = accuracy_score(classified, label )
print('\n test(valid) main data accuracy: ' , scores)

server_decision = (classified == label)

np.saved ('server_decision.npy')



#----------------------------------------------------------











