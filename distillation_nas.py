

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:48:03 2020

@author: behnam

In this code the knowledge distillation technique has been applied to the search space of the Neural Architecture Search methoed propoded in:
https://github.com/akwasigroch/NAS_network_morphism

The main.py from https://github.com/akwasigroch/NAS_network_morphism/main.py file has been modified in this code. 
"""


#import select_gpu
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Conv2D, Flatten, BatchNormalization, ReLU, MaxPooling2D, Activation, Add, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, Sequential, model_from_config, load_model
#from tensorflow.python.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from image_preprocessing_ver2 import ImageDataGenerator as ImageDataGenerator_2
from tensorflow.keras.layers import Lambda, concatenate, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
import shutil
import random
import glob
import os
import datetime
import re
from copy import deepcopy
import math
import numpy as np
import json
from math import pi, cos
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"



gpu_index = 0  # Change this to the index of the GPU you want to use (e.g., 0, 1, 2, ...)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_visible_devices(physical_devices[gpu_index], 'GPU')

size = 224
#data_directory = '/Users/behnam/Documents/NAS_Morph/clean-dataset'

h=0;        
bs = 8 #batch_size
img_rows, img_cols = 224, 224; # Resolution of inputs

##########################################    
datagen = ImageDataGenerator_2(
    rescale=1./255.,
    horizontal_flip=True)

# np_load_old = np.load

# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_logits = np.load('./train_logits_server_imagenet_16_120.npy' , allow_pickle= True)[()]
val_logits = np.load('./val_logits_server_imagenet_16_120.npy' , allow_pickle= True)[()]


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


def preprocess_data(Y):
    """
    function that pre-processes the CIFAR10 dataset as per
    densenet model requirements for input images
    labels are one-hot encoded
    """
   
    Y = tf.keras.utils.to_categorical(Y - 1, num_classes=120)
    # Y = tf.keras.utils.to_categorical(Y)
    return  Y

y_train= preprocess_data( y_train)
y_test = preprocess_data(y_test)



traingenerator = datagen.flow(x_train,y_train,logits_dict = train_logits,shuffle=True)

val_datagen = ImageDataGenerator_2(horizontal_flip=True, rescale=1./255.)
validgenerator = val_datagen.flow(x_test,
                                 y_test,logits_dict = val_logits,
                                 shuffle=False)



class LR_Restart(tf.keras.callbacks.Callback):
    '''SGDR learning rate scheduler'''
    def __init__(self, lr_max, lr_min, restart_epoch):
        super(LR_Restart, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.restart_epoch = restart_epoch
        self.cycle = 0

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch % self.restart_epoch) == 0:
            self.cycle = epoch / self.restart_epoch
        self.curr_epoch = epoch - (self.cycle * self.restart_epoch)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        curr_epoch = self.curr_epoch + (batch/self.params['steps'])
        lr = self.lr_min + (0.5 * (self.lr_max - self.lr_min) * (1 + cos((curr_epoch / self.restart_epoch) * pi)))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class Model_():
    '''Class that allows modification of model with the use of the network morhism operators
    It contains usualy contains two models: teacher - the model before modification, student - the model after modification, that derived
    the weight from teacher'''
    def __init__(self, folder=''):
        self.teacher_json_adres = folder+'model.json'
        self.teacher_json = None
        self.teacher_weights = folder+'weights.h5'
        self.student_json = None
        self.weight_decay = 0.0005

    def create_initial_network(self, epochs=2):
        input_layer = Input(shape=(16, 16, 3))

        # resizing_layer = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_layer)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay))(input_layer)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        #x = MaxPooling2D(pool_size=(1, 1))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        #x = MaxPooling2D(pool_size=(1, 1))(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Flatten()(x)
        # x =  Dense(128, name = 'dense_1', kernel_regularizer=l2(self.weight_decay))(x)
        # x = Dropout(0.5)(x)
        
        logits = Dense(120, name = 'logits', kernel_regularizer=l2(self.weight_decay))(x)
        # model = Model(inputs=inputs, outputs=output)
        # # remove softmax
        # model.layers.pop()
        # 
        # # usual probabilities
        # logits = model.layers[-1].output
        temperature = 20.0
        probabilities = Activation('softmax')(logits)
        
        # softed probabilities
        logits_T = Lambda(lambda x: x/temperature)(logits)
        probabilities_T = Activation('softmax')(logits_T)

        output = concatenate([probabilities, probabilities_T])
        model = Model(inputs=input_layer, outputs=output)
        model.summary()
        
        # model  =  InceptionResnetV2Scratch((16,16,3) , 120 )
        
        #model = Model(inputs=inputs, outputs=output)
        sgdr = LR_Restart(0.001, 1e-5, 10)
        sgd = SGD(lr=0.001, momentum=0.9, nesterov=False)
       
        def knowledge_distillation_loss(y_true, y_pred):    
            temperature = 20
            # split in 
            #    onehot hard true targets
            #    logits from xception
            y_true, logits = y_true[:, :120], y_true[:, 120:]
            
            # convert logits to soft targets
            y_soft = K.softmax(logits/temperature)
            
            # split in 
            #    usual output probabilities
            #    probabilities made softer with temperature
            y_pred, y_pred_soft = y_pred[:, :120], y_pred[:, 120:]    
            lambda_const = 0.05
            return lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)
        def accuracy(y_true, y_pred):
            y_true = y_true[:, :120]
            y_pred = y_pred[:, :120]
            return categorical_accuracy(y_true, y_pred)
        def top_5_accuracy(y_true, y_pred):
            y_true = y_true[:, :120]
            y_pred = y_pred[:, :120]
            return top_k_categorical_accuracy(y_true, y_pred)
        def categorical_crossentropy(y_true, y_pred):
            y_true = y_true[:, :120]
            y_pred = y_pred[:, :120]
            return logloss(y_true, y_pred)
        def soft_logloss(y_true, y_pred):  
            temperature = 20
            logits = y_true[:, 120:]
            y_soft = K.softmax(logits/temperature)
            y_pred_soft = y_pred[:, 120:]    
            return logloss(y_soft, y_pred_soft)
        model.compile(
            sgd, 
            loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred), 
            metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss, knowledge_distillation_loss]
        )


        model.fit(
            traingenerator, 
            steps_per_epoch=4553, epochs=epochs, verbose= 2,
            callbacks=[sgdr],
            validation_data=validgenerator, validation_steps=80, workers=4
        )
        # model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        # #plot_model(model, to_file='model.png')
        # model.fit_generator(traingenerator, steps_per_epoch=1000, epochs=epochs, validation_data=validgenerator,
        #                     max_queue_size=100, workers=4, verbose=1, use_multiprocessing=True, callbacks=[sgdr])

        json_txt = model.to_json()
        json_object = json.loads(json_txt)
        with open('model_start.json', 'w') as f:
            json.dump(json_object, f, indent=2)
        with open(self.teacher_json_adres, 'w') as f:
            json.dump(json_object, f, indent=2)

        model.save_weights('weights_start.h5')
        model.save_weights(self.teacher_weights)

        self.teacher_json = json_object
        self.teacher = model
        self.student = None

    def load_teacher(self, model_name='model_start.json', weights_name='weights_start.h5'):
        '''Load the teacher model from file'''
        with open(model_name, 'r') as f:
            self.teacher_json = json.load(f)
        self.teacher = model_from_config(self.teacher_json)
        self.teacher_json = json.loads(self.teacher.to_json())
        self.teacher.load_weights(weights_name)
        with open(self.teacher_json_adres, 'w') as f:
            json.dump(self.teacher_json, f, indent=2)
        self.teacher.save_weights(self.teacher_weights)

    def wider2net_conv2d(self, layer_index, change_teacher_=False, new_width=None):

        '''Function that add filters to convolutional filter. If new_width is not provided it double numbers of filters'''
        assert self.teacher_json['config']['layers'][layer_index][
            'class_name'] == 'Conv2D', 'Layer is not convolutional'
        self.student_json = deepcopy(self.teacher_json)
        self.layer_list = self.get_layer_list()
        self.layer_list = np.array(self.layer_list)
        bn_index = np.where(self.layer_list[:, 1] == self.layer_list[layer_index, 0])
        relu_index = np.where(self.layer_list[:, 1] == self.layer_list[bn_index[0].item(), 0])
        next_conv_index = np.where(self.layer_list[:, 1] == self.layer_list[relu_index[0].item(), 0])
        assert len(next_conv_index[0]) == 1, 'Wrong place for add'
        assert 'lambda' not in self.layer_list[next_conv_index[0].item(), 0], 'Wider inside add or concatenate block'
        assert 'flatten' not in self.layer_list[next_conv_index[0].item(), 0], 'Last convolutional layer'
        nex_bn_index = np.where(self.layer_list[:, 1] == self.layer_list[next_conv_index[0].item(), 0])
        if 'max' in self.layer_list[next_conv_index[0].item(), 0]:
            next_conv_index = np.where(self.layer_list[:, 1] == self.layer_list[next_conv_index[0].item(), 0])
            assert len(next_conv_index[0]) == 1, 'Wrong place for add'
            nex_bn_index = np.where(self.layer_list[:, 1] == self.layer_list[next_conv_index[0].item(), 0])
        teacher_w1, teacher_b1 = self.teacher.layers[layer_index].get_weights()
        alpha, beta, mean, std = self.teacher.layers[bn_index[0].item()].get_weights()
        teacher_w2, teacher_b2 = self.teacher.layers[next_conv_index[0].item()].get_weights()
        if new_width == None:
            new_width = self.student_json['config']['layers'][layer_index]['config']['filters'] * 2
        n = new_width - teacher_w1.shape[3]
        assert n > 0, "New width smaller than teacher width"
        index = np.random.randint(teacher_w1.shape[3], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, :, :, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
        new_alpha = alpha[index]
        new_beta = beta[index]
        new_mean = mean[index]
        new_std = std[index]
        new_w1 = new_w1+np.random.normal(scale=new_w1.std()*0.05, size=new_w1.shape)
        student_w1 = np.concatenate((teacher_w1, new_w1), axis=3)
        alpha = np.concatenate((alpha, new_alpha))
        beta = np.concatenate((beta, new_beta))
        mean = np.concatenate((mean, new_mean))
        std = np.concatenate((std, new_std))
        new_w2 = new_w2+np.random.normal(scale=new_w2.std()*0.05, size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=2)
        student_w2[:, :, index, :] = new_w2
        student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)
        conv1 = self.find_layer_name('conv2d')[0]
        bn1 = self.find_layer_name('batch_normalization')[0]
        self.student_json = deepcopy(self.teacher_json)
        self.student_json['config']['layers'][layer_index]['config']['filters'] = new_width
        self.student_json['config']['layers'][layer_index]['config']['name'] = conv1
        self.student_json['config']['layers'][layer_index]['name'] = conv1
        self.student_json['config']['layers'][bn_index[0].item()]['config']['name'] = bn1
        self.student_json['config']['layers'][bn_index[0].item()]['name'] = bn1
        self.student_json['config']['layers'][bn_index[0].item()]['inbound_nodes'][0][0][0] = conv1
        self.student_json['config']['layers'][relu_index[0].item()]['inbound_nodes'][0][0][0] = bn1
        conv2 = conv1 = self.find_layer_name('conv2d')[0]
        self.student_json['config']['layers'][next_conv_index[0].item()]['config']['name'] = conv2
        self.student_json['config']['layers'][next_conv_index[0].item()]['name'] = conv2
        self.student_json['config']['layers'][nex_bn_index[0].item()]['inbound_nodes'][0][0][0] = conv2
        tf.keras.backend.clear_session()
        self.student = model_from_config(self.student_json)
        self.student.load_weights(self.teacher_weights, by_name=True)
        self.student.layers[layer_index].set_weights((student_w1, student_b1))
        self.student.layers[bn_index[0].item()].set_weights((alpha, beta, mean, std))
        self.student.layers[next_conv_index[0].item()].set_weights((student_w2, teacher_b2))
        if change_teacher_:
            self.change_teacher()

    def wider2net_conv2d_fc(self, layer_index, change_teacher_=False, new_width=None):

        '''Add filters to the convolutional layer that is placed before fully connected layer'''
        assert self.teacher_json['config']['layers'][layer_index][
            'class_name'] == 'Conv2D', 'Layer is not convolutional'
        self.student_json = deepcopy(self.teacher_json)
        self.layer_list = self.get_layer_list()
        self.layer_list = np.array(self.layer_list)

        bn_index = np.where(self.layer_list[:, 1] == self.layer_list[layer_index, 0])

        relu_index = np.where(self.layer_list[:, 1] == self.layer_list[bn_index[0], 0])
        flatten_index = np.where(self.layer_list[:, 1] == self.layer_list[relu_index[0], 0])
        fc1_index = np.where(self.layer_list[:, 1] == self.layer_list[flatten_index[0], 0])

        assert self.teacher_json['config']['layers'][fc1_index[0].item()][
            'class_name'] == 'Dense', 'Next layer is not Dense'

        teacher_w1, teacher_b1 = self.teacher.layers[layer_index].get_weights()
        alpha, beta, mean, std = self.teacher.layers[bn_index[0].item()].get_weights()
        teacher_w2, teacher_b2 = self.teacher.layers[fc1_index[0].item()].get_weights()
        if new_width == None:
            new_width = self.student_json['config']['layers'][layer_index]['config']['filters'] * 2
        n = new_width - teacher_w1.shape[3]
        assert n > 0, "New width smaller than teacher width"

        index = np.random.randint(teacher_w1.shape[3], size=n)

        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, :, :, index]
        new_b1 = teacher_b1[index]


        new_w2 = np.copy(teacher_w2)
        new_w2_split = new_w2[0::teacher_w1.shape[3], :]
        new_w2_split = np.expand_dims(new_w2_split, 2)
        for i in range(1, teacher_w1.shape[3]):
            new_w2_split = np.concatenate((new_w2_split, new_w2[i::teacher_w1.shape[3], :, np.newaxis]), axis=2)

        new_w2 = new_w2_split[:, :, index] / factors.reshape((1, 1, -1))

        new_alpha = alpha[index]
        new_beta = beta[index]
        new_mean = mean[index]
        new_std = std[index]

        new_w1 = new_w1+np.random.normal(scale=new_w1.std()*0.05, size=new_w1.shape)
        student_w1 = np.concatenate((teacher_w1, new_w1), axis=3)

        alpha = np.concatenate((alpha, new_alpha))
        beta = np.concatenate((beta, new_beta))
        mean = np.concatenate((mean, new_mean))
        std = np.concatenate((std, new_std))

        new_w2 = new_w2+np.random.normal(scale=new_w2.std()*0.05, size=new_w2.shape)

        student_w2 = np.concatenate((new_w2_split, new_w2), axis=2)

        student_w2[:, :, index] = new_w2

        student_w2 = np.swapaxes(student_w2, 1, 2)

        student_w2 = np.reshape(student_w2, (-1, teacher_w2.shape[1]))

        student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)
        self.student_json = deepcopy(self.teacher_json)
   
        conv = self.find_layer_name('conv2d', teacher=False)[0]
        bn1 = self.find_layer_name('batch_normalization')[0]
        dense = self.find_layer_name('dense')[0]

        self.student_json['config']['layers'][layer_index]['config']['filters'] = new_width

        self.student_json['config']['layers'][layer_index]['config']['name'] = conv
        self.student_json['config']['layers'][layer_index]['name'] = conv

        self.student_json['config']['layers'][bn_index[0].item()]['config']['name'] = bn1
        self.student_json['config']['layers'][bn_index[0].item()]['name'] = bn1
        self.student_json['config']['layers'][bn_index[0].item()]['inbound_nodes'][0][0][0] = conv

        self.student_json['config']['layers'][relu_index[0].item()]['inbound_nodes'][0][0][0] = bn1

        self.student_json['config']['layers'][fc1_index[0].item()]['config']['name'] = dense
        self.student_json['config']['layers'][fc1_index[0].item()]['name'] = dense

        self.student_json['config']['output_layers'][0][0] = dense
        tf.keras.backend.clear_session()
        self.student = model_from_config(self.student_json)
        self.student.load_weights(self.teacher_weights, by_name=True)

        self.student.layers[layer_index].set_weights((student_w1, student_b1))
        self.student.layers[bn_index[0].item()].set_weights((alpha, beta, mean, std))
        self.student.layers[fc1_index[0].item()].set_weights((student_w2, teacher_b2))
        if change_teacher_:
            self.change_teacher()

    def deeper2net_conv2d(self, index, change_teacher_=False):

        '''Add convolutional layer after convolutional layer'''

        assert self.teacher_json['config']['layers'][index]['class_name'] == 'Conv2D', 'Layer is not convolutional'
        teacher_w = self.teacher.layers[index].get_weights()[0]
        kh, kw, _, filters = teacher_w.shape
        student_w = np.zeros((kh, kw, filters, filters))
        for i in range(filters):
            student_w[(kh - 1) // 2, (kw - 1) // 2, i, i] = 1.
        self.student_json = deepcopy(self.teacher_json)
        student_b = np.zeros(filters)

        self.layer_list = self.get_layer_list()
        self.layer_list = np.array(self.layer_list)
        bn_index = np.where(self.layer_list[:, 1] == self.layer_list[index, 0])

        for i in range(len(bn_index[0])):

            if 'lambda' in self.layer_list[bn_index[0][i]][0]:
                a = np.delete(bn_index[0], i)
                bn_index = (a,)
                break

        relu_index = np.where(self.layer_list[:, 1] == self.layer_list[bn_index[0], 0])

        conv_or_maxpool_index = self.return_next_layer(relu_index[0].item())

        a = np.random.randint(0, 10e6, 1)
        conv1 = self.find_layer_name('conv2d')[0]
        self.student_json['config']['layers'].append({'class_name': 'Conv2D',
                                                      'config': tf.keras.layers.Conv2D(filters, (3, 3),
                                                                                       input_shape=(28, 28, filters),
                                                                                       padding='same', kernel_regularizer=l2(self.weight_decay)).get_config()})
        self.student_json['config']['layers'][-1]['name'] = conv1
        self.student_json['config']['layers'][-1]['config']['name'] = conv1
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][index]['inbound_nodes'])
        self.student_json['config']['layers'][-1]['inbound_nodes'][0][0][0] = deepcopy(
            self.student_json['config']['layers'][relu_index[0].item()]['name'])
        bn1 = self.find_layer_name('batch_normalization')[0]
        self.student_json['config']['layers'].append({'class_name': 'BatchNormalization',
                                                      'config': tf.keras.layers.BatchNormalization(
                                                          name=bn1).get_config()})
        self.student_json['config']['layers'][-1]['config']['epsilon'] = 1e-10
        self.student_json['config']['layers'][-1]['name'] = bn1

        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][index]['inbound_nodes'])
        self.student_json['config']['layers'][-1]['inbound_nodes'][0][0][0] = deepcopy(
            self.student_json['config']['layers'][-2]['name'])
        relu1 = self.find_layer_name('re_lu')[0]
        self.student_json['config']['layers'].append(
            {'class_name': 'ReLU', 'config': tf.keras.layers.ReLU(name=relu1).get_config()})
        self.student_json['config']['layers'][-1]['name'] = relu1
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][index]['inbound_nodes'])
        self.student_json['config']['layers'][-1]['inbound_nodes'][0][0][0] = deepcopy(
            self.student_json['config']['layers'][-2]['name'])

        layer_list = self.get_layer_list_with_details(teacher=True)
        relu_name = layer_list[relu_index[0].item()][0]

        for item in conv_or_maxpool_index:
            for element in self.student_json['config']['layers'][item]['inbound_nodes'][0]:
                if element[0] == relu_name:
                    element[0] = deepcopy(self.student_json['config']['layers'][-1]['name'])

        tf.keras.backend.clear_session()
        self.student = model_from_config(self.student_json)

        self.student.load_weights(self.teacher_weights, by_name=True)

        nazwa = conv1
        student_w = student_w + np.random.normal(scale=student_w.std()*0.01, size=student_w.shape)
        self.student.get_layer(nazwa).set_weights((student_w, student_b))
        if change_teacher_:
            self.change_teacher()

    def wider2net_fc(self, new_width, layer_index, change_teacher_=False):

        '''Expand fully connected layer'''
        assert self.teacher_json['config']['layers'][layer_index]['class_name'] == 'Dense', 'Layer is not Dense'

        self.student_json = deepcopy(self.teacher_json)
        self.layer_list = self.get_layer_list()
        self.layer_list = np.array(self.layer_list)
        next_dense_index = np.where(self.layer_list[:, 1] == self.layer_list[layer_index, 0])

        teacher_w1, teacher_b1 = self.teacher.layers[layer_index].get_weights()
        teacher_w2, teacher_b2 = self.teacher.layers[next_dense_index[0].item()].get_weights()
        assert teacher_w1.shape[1] == teacher_w2.shape[0], (
            'successive layers from teacher model should have compatible shapes')
        assert teacher_w1.shape[1] == teacher_b1.shape[0], (
            'weight and bias from same layer should have compatible shapes')
        assert new_width > teacher_w1.shape[1], (
            'new width (nout) should be bigger than the existing one')

        n = new_width - teacher_w1.shape[1]

        index = np.random.randint(teacher_w1.shape[1], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[index, :] / factors[:, np.newaxis]

        student_w1 = np.concatenate((teacher_w1, new_w1), axis=1)
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=0)
        student_w2[index, :] = new_w2
        student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

        self.student_json = deepcopy(self.teacher_json)
        a = np.random.randint(0, 10e6, 1)

        self.student_json['config']['layers'][layer_index]['config']['units'] = new_width
        self.student_json['config']['layers'][layer_index]['config']['name'] = 'dense_%d' % (a[0])
        self.student_json['config']['layers'][layer_index]['name'] = 'dense_%d' % (a[0])

        self.student_json['config']['layers'][next_dense_index[0].item()]['inbound_nodes'][0][0][0] = 'dense_%d' % (
            a[0])
        self.student_json['config']['layers'][next_dense_index[0].item()]['config']['name'] = 'dense_%d' % (a[0] + 1)
        self.student_json['config']['layers'][next_dense_index[0].item()]['name'] = 'dense_%d' % (a[0] + 1)

        self.student_json['config']['output_layers'][0][0] = 'dense_%d' % (a[0] + 1)

        tf.keras.backend.clear_session()
        self.student = model_from_config(self.student_json)
        self.student.load_weights(self.teacher_weights, by_name=True)

        self.student.get_layer('dense_%d' % (a[0])).set_weights((student_w1, student_b1))
        self.student.get_layer('dense_%d' % (a[0] + 1)).set_weights((student_w2, teacher_b2))
        if change_teacher_:
            self.change_teacher()

        return student_w1, student_b1, student_w2

    def add(self, layer_index, change_teacher_=False):
        

        '''Create 'add motif' as in https://arxiv.org/pdf/1806.02639.pdf '''
        self.student_json = deepcopy(self.teacher_json)

        self.layer_list = self.get_layer_list()
        self.layer_list = np.array(self.layer_list)

        conv_index = np.atleast_1d(layer_index)
        conv_index = (conv_index)

        assert 'conv' in self.layer_list[conv_index.item(), 0], 'Wrong layer index'

        bn_index = np.where(self.layer_list[:, 1] == self.layer_list[conv_index[0], 0])

        relu_index = np.where(self.layer_list[:, 1] == self.layer_list[bn_index[0], 0])
        next_conv_index = self.return_next_layer(relu_index[0].item())

        lambda1 = self.find_layer_name('lambda')[0]
        self.student_json['config']['layers'].append(
            {'class_name': 'Lambda', 'config': tf.keras.layers.Lambda(lambda x: x * 0.5).get_config()})
        self.student_json['config']['layers'][-1]['name'] = lambda1
        self.student_json['config']['layers'][-1]['config']['name'] = lambda1
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][-6]['inbound_nodes'])
        self.set_inbound_node(-1, self.layer_list[relu_index[0].item(), 0])

        conv1 = self.find_layer_name('conv2d')[0]
        bn1 = self.find_layer_name('batch_normalization')[0]
        relu1 = self.find_layer_name('re_lu')[0]
        self.student_json['config']['layers'].append(
            deepcopy(self.student_json['config']['layers'][conv_index[0].item()]))
        self.set_name(-1, conv1)

        self.student_json['config']['layers'].append(
            deepcopy(self.student_json['config']['layers'][bn_index[0].item()]))
        self.set_name(-1, bn1)
        self.set_inbound_node(-1, conv1)

        self.student_json['config']['layers'].append(
            deepcopy(self.student_json['config']['layers'][relu_index[0].item()]))
        self.set_name(-1, relu1)
        self.set_inbound_node(-1, bn1)
        lambda2 = self.find_layer_name('lambda')[0]
        self.student_json['config']['layers'].append(deepcopy(self.student_json['config']['layers'][-4]))
        self.set_name(-1, lambda2)
        self.set_inbound_node(-1, relu1)

        if layer_index == 0:
            layer_index += 1
        add1 = self.find_layer_name('add')[0]
        self.student_json['config']['layers'].append(
            {'class_name': 'Add', 'config': tf.keras.layers.Add().get_config()})
        self.set_name(-1, add1)
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][layer_index]['inbound_nodes'])
        self.set_inbound_node(-1, lambda1)
        self.student_json['config']['layers'][-1]['inbound_nodes'][0].append(
            deepcopy(self.student_json['config']['layers'][layer_index]['inbound_nodes'][0][0]))
        self.student_json['config']['layers'][-1]['inbound_nodes'][0][1][0] = lambda2

        layer_list = self.get_layer_list_with_details()
        relu_name = layer_list[relu_index[0].item()][0]
        for index in next_conv_index:
            for element in self.student_json['config']['layers'][index]['inbound_nodes'][0]:
                if element[0] == relu_name:
                    element[0] = add1
        tf.keras.backend.clear_session()
        
        
        self.student = model_from_config(self.student_json)
        
        self.student.load_weights(self.teacher_weights, by_name=True)

        weights = self.student.get_layer(self.layer_list[conv_index[0].item(), 0]).get_weights()
        noise = np.random.normal(0, 0.01*weights[0].std(), size=weights[0].shape)
        weightsnoise = weights[0]+noise
        self.student.get_layer(conv1).set_weights((weightsnoise, weights[1]))
        self.student.get_layer(bn1).set_weights(
            self.student.get_layer(self.layer_list[bn_index[0].item(), 0]).get_weights())
        self.student_json = json.loads(self.student.to_json())

        if change_teacher_:
            self.change_teacher()

    def concat(self, layer_index, change_teacher_=False):
        

        '''Create 'concatenation motif' as in https://arxiv.org/pdf/1806.02639.pdf'''
        self.student_json = deepcopy(self.teacher_json)

        self.layer_list = self.get_layer_list()
        self.layer_list = np.array(self.layer_list)
        conv_index = np.atleast_1d(layer_index)
        conv_index = (conv_index)
        assert 'conv' in self.layer_list[conv_index.item(), 0], 'Wrong layer index'

        bn_index = np.where(self.layer_list[:, 1] == self.layer_list[conv_index[0], 0])
        relu_index = np.where(self.layer_list[:, 1] == self.layer_list[bn_index[0], 0])
        next_conv_index = self.return_next_layer(relu_index[0].item())
        conv_weights = self.teacher.get_layer(self.layer_list[conv_index[0].item(), 0]).get_weights()
        bn_weights = self.teacher.get_layer(self.layer_list[bn_index[0].item(), 0]).get_weights()
        filters = conv_weights[0].shape[3]

        self.student_json['config']['layers'][conv_index[0].item()]['config']['filters'] = int(filters/2)

        conv1 = self.find_layer_name('conv2d')[0]
        bn1 = self.find_layer_name('batch_normalization')[0]
        self.set_name(conv_index[0].item(), conv1)
        self.set_name(bn_index[0].item(), bn1)
        self.set_inbound_node(bn_index[0].item(), conv1)

        self.set_inbound_node(relu_index[0].item(), bn1)

        conv2 = self.find_layer_name('conv2d')[0]
        bn2 = self.find_layer_name('batch_normalization')[0]
        relu2 = self.find_layer_name('re_lu')[0]
        self.student_json['config']['layers'].append(
            deepcopy(self.student_json['config']['layers'][conv_index[0].item()]))
        self.set_name(-1, conv2)

        self.student_json['config']['layers'].append(
            deepcopy(self.student_json['config']['layers'][bn_index[0].item()]))
        self.set_name(-1, bn2)
        self.set_inbound_node(-1, conv2)

        self.student_json['config']['layers'].append(
            deepcopy(self.student_json['config']['layers'][relu_index[0].item()]))
        self.set_name(-1, relu2)
        self.set_inbound_node(-1, bn2)

        if layer_index == 0:
            layer_index += 1
        concatenate1 = self.find_layer_name('concatenate')[0]
        self.student_json['config']['layers'].append(
            {'class_name': 'Concatenate', 'config': tf.keras.layers.Concatenate().get_config()})
        self.set_name(-1, concatenate1)
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][layer_index]['inbound_nodes'])
        self.set_inbound_node(-1, self.layer_list[relu_index[0].item(), 0])
        self.student_json['config']['layers'][-1]['inbound_nodes'][0].append(
            deepcopy(self.student_json['config']['layers'][layer_index]['inbound_nodes'][0][0]))
        self.student_json['config']['layers'][-1]['inbound_nodes'][0][1][0] = relu2
        layer_list = self.get_layer_list_with_details()
        relu_name = layer_list[relu_index[0].item()][0]

        if 'concatenate' in layer_list[next_conv_index[0]][0]:
            for element in self.student_json['config']['layers'][next_conv_index[0]]['inbound_nodes'][0]:

                if element[0] == relu_name:
                    element[0] = concatenate1
        else:
            for index in next_conv_index:

                self.set_inbound_node(index, concatenate1)


        tf.keras.backend.clear_session()
        self.student = model_from_config(self.student_json)

        self.student.load_weights(self.teacher_weights, by_name=True)

        self.student.get_layer(conv1).set_weights(
            (conv_weights[0][:, :, :, 0:int(filters/2)], conv_weights[1][0:int(filters/2)]))
        self.student.get_layer(conv2).set_weights(
            (conv_weights[0][:, :, :, int(filters/2):], conv_weights[1][int(filters/2):]))

        self.student.get_layer(bn1).set_weights([item[0:int(filters/2)] for item in bn_weights])
        self.student.get_layer(bn2).set_weights([item[int(filters/2):] for item in bn_weights])

        self.student_json = json.loads(self.student.to_json())

        if change_teacher_:
            self.change_teacher()

    def skip(self, layer_index, change_teacher_=False):
        '''Add skip conection. This is combination of 'add' and 'deeper2net_conv2d' functions'''

        a = self.get_layer_list(teacher=True)
        a = [item[0] for item in a]
        
        self.deeper2net_conv2d(layer_index, change_teacher_=True)
        
        b = self.get_layer_list(teacher=True)
        b = [item[0] for item in b]
        
        difference = list(set(b)-set(a))
        last_layer = [x for x in difference if 're_lu' in x]
        conv_layer = [x for x in difference if 'conv' in x]
        
        self.student_json = deepcopy(self.teacher_json)
        self.layer_list = self.get_layer_list()
        self.layer_list = np.array(self.layer_list)
    
        lambda1 = self.find_layer_name('lambda', teacher=False)[0]
        self.student_json['config']['layers'].append(
            {'class_name': 'Lambda', 'config': tf.keras.layers.Lambda(lambda x: x * 0.5).get_config()})
        self.student_json['config']['layers'][-1]['name'] = lambda1
        self.student_json['config']['layers'][-1]['config']['name'] = lambda1
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][-6]['inbound_nodes'])
        self.set_inbound_node(-1, last_layer[0])

        a = np.random.randint(0, 10e6, 1)
        lambda2 = self.find_layer_name('lambda', teacher=False)[0]
        self.student_json['config']['layers'].append(
            {'class_name': 'Lambda', 'config': tf.keras.layers.Lambda(lambda x: x * 0.5).get_config()})
        self.student_json['config']['layers'][-1]['name'] = lambda2
        self.student_json['config']['layers'][-1]['config']['name'] = lambda2
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][-2]['inbound_nodes'])
        self.set_inbound_node(-1, conv_layer[0])

        add1 = self.find_layer_name('add')[0]
        self.student_json['config']['layers'].append(
            {'class_name': 'Add', 'config': tf.keras.layers.Add().get_config()})
        self.set_name(-1, add1)
        self.student_json['config']['layers'][-1]['inbound_nodes'] = deepcopy(
            self.student_json['config']['layers'][layer_index]['inbound_nodes'])
        self.set_inbound_node(-1, lambda1)
        self.student_json['config']['layers'][-1]['inbound_nodes'][0].append(
            deepcopy(self.student_json['config']['layers'][layer_index]['inbound_nodes'][0][0]))
        self.student_json['config']['layers'][-1]['inbound_nodes'][0][1][0] = lambda2
        layer_list = self.get_layer_list_with_details(teacher=True)
        conv_layer_index = self.return_layer_index(conv_layer[0])
        bn_index = self.return_next_layer(conv_layer_index)
        relu_index = self.return_next_layer(bn_index[0])
        relu_name = layer_list[relu_index[0]][0]
        next_layer_index = self.return_next_layer(relu_index[0])
        for item in next_layer_index:
            for element in self.student_json['config']['layers'][item]['inbound_nodes'][0]:
                if element[0] == relu_name:
                    element[0] = add1
        
        
        tf.keras.backend.clear_session()
        self.student = model_from_config(self.student_json)
        self.student.load_weights(self.teacher_weights, by_name=True)
        if change_teacher_:
            self.change_teacher()
        
    def get_layer_list(self, teacher=False):
        '''Return list of layers with following layers'''
        if teacher == False:
            layer_list = []
            for i in range(len(self.student_json['config']['layers'])):

                if i == 0:
                    layer_list.append([self.student_json['config']['layers'][0]['name'], 'input_layer'])
                else:
                    layer_list.append([self.student_json['config']['layers'][i]['name'],
                                       self.student_json['config']['layers'][i]['inbound_nodes'][0][0][0]])

        if teacher:
            layer_list = []
            for i in range(len(self.teacher_json['config']['layers'])):

                if i == 0:
                    layer_list.append([self.teacher_json['config']['layers'][i]['name'], 'input_layer'])
                else:
                    layer_list.append([self.teacher_json['config']['layers'][i]['name'],
                                       self.teacher_json['config']['layers'][i]['inbound_nodes'][0][0][0]])
        return layer_list
    def get_layer_list_with_details(self, teacher=False):
        '''Return list of layers'''
        if teacher == False:
            layer_list = []
            for i in range(len(self.student_json['config']['layers'])):

                if i == 0:
                    layer_list.append([self.student_json['config']['layers'][0]['name'], 'input_layer'])
                else:
                    name = [self.student_json['config']['layers'][i]['name']]
                    inputs = [element[0] for element in self.student_json['config']['layers'][i]['inbound_nodes'][0]]
                    name.extend(inputs)
                    layer_list.append(name)

        if teacher:
            layer_list = []
            for i in range(len(self.teacher_json['config']['layers'])):

                if i == 0:
                    layer_list.append([self.teacher_json['config']['layers'][i]['name'], 'input_layer'])
                else:
                    name = [self.teacher_json['config']['layers'][i]['name']]
                    inputs = [element[0] for element in self.teacher_json['config']['layers'][i]['inbound_nodes'][0]]
                    name.extend(inputs)
                    layer_list.append(name)

        return layer_list

    def set_name(self, layer_index, name):
        '''Set name of layer'''
        self.student_json['config']['layers'][layer_index]['config']['name'] = name
        self.student_json['config']['layers'][layer_index]['name'] = name

    def set_inbound_node(self, layer_index, inbound_node):
        '''Set inbound node of layer'''
        self.student_json['config']['layers'][layer_index]['inbound_nodes'][0][0][0] = inbound_node

    def change_teacher(self):
        '''Make student network teacher network'''
        self.student.save_weights(self.teacher_weights)
        tf.keras.backend.clear_session()
        self.teacher = model_from_config(self.student_json)
        self.teacher.load_weights(self.teacher_weights, by_name=True)
        self.teacher_json = json.loads(self.teacher.to_json())
        with open(self.teacher_json_adres, 'w') as f:
            json.dump(self.teacher_json, f, indent=2)

    def evaluate(self, teacher=True, print_result=False):
        sgd = SGD(momentum=0.9, nesterov=False)

        self.teacher.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        result = self.teacher.evaluate_generator(validgenerator)
        if print_result:
            print(result)
        return result

    def return_available_modyfications(self):
        wider2net_conv2d = []
        deeper2net_conv2d = []
        wider2net_conv2d_fc = []
        add = []
        concat = []
        skip = []

        layer_list = self.get_layer_list(teacher=True)
        proper_layer_list = self.get_layer_list_with_details(teacher=True)
        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            second = self.return_next_layer(i)
            if len(second) > 1:
                continue

            third = self.return_next_layer(second[0]) 
            fourth = self.return_next_layer(third[0])

            if len(fourth) > 1:
                continue
            if len(proper_layer_list[fourth[0]][1:]) > 1:
                continue
            if 'flatten' in layer_list[fourth[0]][0]:
                continue
            if 'lambda' in layer_list[fourth[0]][0]:
                continue
            if 'conv' or 'max' in layer_list[fourth[0]][0]:
                fifth = self.return_next_layer(fourth[0])
                if len(fifth) > 1:
                    continue
            wider2net_conv2d.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            second = self.return_next_layer(i)  
            third = self.return_next_layer(second[0]) 
            fourth = self.return_next_layer(third[0])  
            if 'flatten' in layer_list[fourth[0]]:
                wider2net_conv2d_fc.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' in element[0]:
                deeper2net_conv2d.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            next_layer = self.return_next_layer(i)
            if len(next_layer) > 1:
                continue
            skip.append(i)

        for i, element in enumerate(layer_list):
            if 'conv' not in element[0]:
                continue
            next_layer = self.return_next_layer(i)
            if len(next_layer) > 1:
                continue
            add.append(i)
            concat.append(i)

        available = {'wider2net_conv2d': wider2net_conv2d,
                     'deeper2net_conv2d': deeper2net_conv2d, 'add': add, 'concat': concat, 'skip': skip}
        # available = {'wider2net_conv2d': wider2net_conv2d,
        #              'deeper2net_conv2d': deeper2net_conv2d}
        return available

    

    def return_next_layer(self, layer_index):
        layer_list = self.get_layer_list_with_details(teacher=True)
        next_layers = []
        for i in range(1, len(layer_list)):
            if layer_list[layer_index][0] in layer_list[i][1:]:
                next_layers.append(i)
        return list(next_layers)

    def print_layer_name(self, index):
        layer_list = self.get_layer_list_with_details(teacher=True)
        print(layer_list[index][0])

    def return_layer_index(self, name):
        layer_list = self.get_layer_list_with_details(teacher=True)
        for i, element in enumerate(layer_list):
            if element[0] == name:
                return i

    def find_layer_name(self, name, teacher=False):
        if teacher:
            all_layer_list = self.get_layer_list_with_details(teacher=True)
        else:
            all_layer_list = self.get_layer_list_with_details(teacher=False)

        layer_list = []
        for element in all_layer_list:
            if name in element[0]:
                pattern = '_[0-9]'
                position = re.search(pattern, element[0])
                if position != None:
                    position = position.start()
                    position += 1
                    layer_list.append(int(element[0][position:]))

        if len(layer_list) == 0:
            return (name+'_0', 0)
        else:
            return (name+'_'+str(max(layer_list)+1), 0)
        
   


    def train(self, epochs=25, lr=0.005, verbose=2, checkpoint_cond=False):
        sgd = SGD(lr=lr)
        sgdr = LR_Restart(lr, 0, epochs)
        def knowledge_distillation_loss(y_true, y_pred):    
            temperature = 20
            # split in 
            #    onehot hard true targets
            #    logits from xception
            y_true, logits = y_true[:, :120], y_true[:, 120:]
            
            # convert logits to soft targets
            y_soft = K.softmax(logits/temperature)
            
            # split in 
            #    usual output probabilities
            #    probabilities made softer with temperature
            y_pred, y_pred_soft = y_pred[:, :120], y_pred[:, 120:]    
            lambda_const = 0.05
            return lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)
        def accuracy(y_true, y_pred):
            y_true = y_true[:, :120]
            y_pred = y_pred[:, :120]
            return categorical_accuracy(y_true, y_pred)
        def top_5_accuracy(y_true, y_pred):
            y_true = y_true[:, :120]
            y_pred = y_pred[:, :120]
            return top_k_categorical_accuracy(y_true, y_pred)
        def categorical_crossentropy(y_true, y_pred):
            y_true = y_true[:, :120]
            y_pred = y_pred[:, :120]
            return logloss(y_true, y_pred)
        def soft_logloss(y_true, y_pred):  
            temperature = 20
            logits = y_true[:, 120:]
            y_soft = K.softmax(logits/temperature)
            y_pred_soft = y_pred[:, 120:]    
            return logloss(y_soft, y_pred_soft)
        
        self.teacher.compile(
            sgd, 
            loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred), 
            metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss, knowledge_distillation_loss]
        )


        # history = self.teacher.fit_generator(
        #     traingenerator, 
        #     steps_per_epoch=1000, epochs=epochs, verbose= verbose,
        #     callbacks=[sgdr], use_multiprocessing=False, max_queue_size=10,
        #     validation_data=validgenerator, validation_steps=80, workers=4
        # )

        #self.teacher.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        filepath = "_epoch_"+"{epoch:02d}"+"_val_loss_"+"{val_loss:.4f}"+"_val_acc_"+"{val_acc:.3f}"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        if checkpoint_cond:
            
            history = self.teacher.fit_generator(
            traingenerator, 
             epochs=epochs, verbose= verbose,
            callbacks=[sgdr , checkpoint], use_multiprocessing=False, max_queue_size=10,
            validation_data=validgenerator, validation_steps=80, workers=4)
            return history
        else:
            history = self.teacher.fit(
            traingenerator, 
            steps_per_epoch=4553, 
             epochs=epochs, verbose= 2,
            callbacks=[sgdr], use_multiprocessing=False, max_queue_size=10,
            validation_data=validgenerator, validation_steps=80, workers=4
            )

            self.teacher.save_weights(self.teacher_weights)
            return history


class Organism():
    def __init__(self, number, epoch=''):
        self.numer = number
        self.folder = epoch+'model'+str(number)+'/'
        if os.path.isdir(self.folder[:-1]):
            shutil.rmtree(self.folder)
            os.mkdir(self.folder)
        else:
            os.mkdir(self.folder)
        self.model = Model_(self.folder)

    def random_modification(self):
        '''Select random modification'''
        available_modifications = self.model.return_available_modyfications()
        while True:
            random_modification = random.choice(list(available_modifications.keys()))
            if len(available_modifications[random_modification]) > 0:
                break
        
        random_index = random.choice(list(available_modifications[random_modification]))
        
        function = getattr(self.model, random_modification)
        
        function(random_index, change_teacher_=True)
        #plot_model(self.model.teacher, to_file=self.folder+'model_after_modification.png')
        
        return random_modification

    def evaluate(self):
        return self.model.evaluate()

    def train(self, epochs=25, lr=0.005, verbose=2):
        return self.model.train(epochs, lr, verbose)

    def plot_model(self):
        #plot_model(self.model.teacher, to_file=self.folder+'model_after_modification.png')
        print('')


class HillClimb():
    def __init__(self, number_of_organism, epochs):
        self.number_of_organism = number_of_organism
        self.epochs = epochs

    def xxstart(self):
        adresy = glob.glob('model*/')
        for adres in adresy:
            shutil.rmtree(adres)
        if os.path.isdir('best'):
            shutil.rmtree('best')
            os.mkdir('best')
        else:
            os.mkdir('best')
        shutil.copyfile('model_start.json', 'best/model_start.json')
        shutil.copyfile('weights_start.h5', 'best/weights_start.h5')

        previous_best = 0
        best_founded = 0
        
        terminate_on_th_edge = False
        for epoch in range(self.epochs):
            if terminate_on_th_edge:
                break
            print('\nEpoch %d' % epoch)
            edge = 0
            list_of_organisms = []
            list_of_result = []
            for i in range(self.number_of_organism):
                list_of_organisms.append(Organism(i))
            for i in range(self.number_of_organism):

                while True:
                    tf.keras.backend.clear_session()
                    print('\nEpoch %d' % epoch)
                    print('\nModel loading %d' % i)
                    list_of_organisms[i].model.load_teacher(
                        model_name='best/model_start.json', weights_name='best/weights_start.h5')
                    print('\nModel loaded %d' % i)
                    current_np = list_of_organisms[i].model.teacher.count_params() 
                    if i == 0:
                        print('Organism 0, no modification')
                        list_of_organisms[i].plot_model()
                        break
                        # list_of_organisms[i].model.teacher.summary()
                    modifications = []
                    number_of_modifications = 2
                    '''Select random modifications'''
                    for _ in range(number_of_modifications):
                        modification = list_of_organisms[i].random_modification()
                        modifications.append(modification)
                    print('Organism %d: modifications: %s' % (i, modifications))
                    if list_of_organisms[i].model.teacher.count_params() < 5000000:
                        print('Number of parameters: %d' % list_of_organisms[i].model.teacher.count_params())
                        break
                    else:
                        if edge > 3:
                            terminate_on_th_edge = True
                            break
                        if current_np > 4800000:
                            edge += 1
                        print('Repeat drawing of network morphism function: %d' % list_of_organisms[i].model.teacher.count_params())
                
                # list_of_organisms[i].model.teacher.summary()
                if i == 0:
                    list_of_result.append(0)
                    print('\nEpoch %d' % epoch)
                    print('Organism %d result: %f' % (i, list_of_result[i]))
                    print('Organism %d np: %f' % (i, list_of_organisms[i].model.teacher.count_params()))
                else: 
                    history = list_of_organisms[i].train(epochs=8, lr=0.001, verbose=1)
                    tf.get_logger().setLevel('ERROR') 
                    organism_result = np.mean(history.history['val_accuracy'][-4:])
                    list_of_result.append(organism_result)
                    print('\nEpoch %d' % epoch)
                    print('Organism %d result: %f' % (i, list_of_result[i]))
                    print('Organism %d np: %f' % (i, list_of_organisms[i].model.teacher.count_params()))
            best = list_of_result.index(max(list_of_result))
            print('\n=============================\nBest: %d, result: %f, previous: %f\n===========================' %
                  (best, list_of_result[best], previous_best))
            shutil.copyfile(list_of_organisms[best].folder+'model.json', 'best/model_start.json')
            shutil.copyfile(list_of_organisms[best].folder+'weights.h5', 'best/weights_start.h5')
            shutil.copyfile(list_of_organisms[best].folder+'model.json', 'best/model_epoch%d.json' % epoch)
            shutil.copyfile(list_of_organisms[best].folder+'weights.h5', 'best/weights_epoch%d.h5' % epoch)

            if os.path.exists(list_of_organisms[best].folder+'model_after_modification.png'):
                shutil.copyfile(list_of_organisms[best].folder+'model_after_modification.png', 'best/model.png')
            print('Algorithm found a new best organism')
            previous_best = max(list_of_result)
            best_founded += 1
            # if previous_best < max(list_of_result):
            #     shutil.copyfile(list_of_organisms[best].folder+'model.json', 'best/model_start.json')
            #     shutil.copyfile(list_of_organisms[best].folder+'weights.h5', 'best/weights_start.h5')
            #     shutil.copyfile(list_of_organisms[best].folder+'model.json', 'best/model_epoch%d.json' % epoch)
            #     shutil.copyfile(list_of_organisms[best].folder+'weights.h5', 'best/weights_epoch%d.h5' % epoch)

            #     if os.path.exists(list_of_organisms[best].folder+'model_after_modification.png'):
            #         shutil.copyfile(list_of_organisms[best].folder+'model_after_modification.png', 'best/model.png')
            #     print('Algorithm found a new best organism')
            #     previous_best = max(list_of_result)
            #     best_founded += 1
            # else:
            #     shutil.copyfile(list_of_organisms[0].folder+'model.json', 'best/model_start.json')
            #     shutil.copyfile(list_of_organisms[0].folder+'weights.h5', 'best/weights_start.h5')
            #     shutil.copyfile(list_of_organisms[0].folder+'model.json', 'best/model_epoch%d.json' % epoch)
            #     shutil.copyfile(list_of_organisms[0].folder+'weights.h5', 'best/weights_epoch%d.h5' % epoch)

            #     if os.path.exists(list_of_organisms[0].folder+'model_after_modification.png'):
            #         shutil.copyfile(list_of_organisms[0].folder+'model_after_modification.png', 'best/model.png')
            #     print('Algorithm did not find a new best organism')
            print('Total new organism: %f' % (best_founded))
            with open('best/results.txt', 'a') as myfile:
                myfile.write(str(datetime.datetime.now()))
                for i in range(self.number_of_organism):
                    myfile.write('Epoch: %d, organism %d accuracy: %f\n' % (epoch, i, list_of_result[i]))
                myfile.write('Epoch: %d, best accuracy: %f\n\n\n' % (epoch, list_of_result[best]))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('ERROR')  # 'INFO', 'WARNING', 'ERROR', or 'FATAL'
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import time
model = Model_()

model.create_initial_network(epochs=30)
evolution = HillClimb(number_of_organism=5, epochs=8)
t = time.time()
evolution.xxstart()
search_time = (time.time() - t)/3600
print('finished searching in {} hours'.format(search_time))
model = Model_()
model.load_teacher(model_name='best/model_start.json', weights_name='best/weights_start.h5')
model.teacher.summary()
history = model.train(epochs=30, lr=0.001, verbose=2)


model_ = model.teacher

model_.layers.pop()
new_model = Model(model_.inputs, model_.layers[-5].output)
new_model.summary()
logits =new_model.layers[-1].output

probabilities = Activation('softmax')(logits)


model_ = Model(new_model.input, probabilities)
model_.summary()

model = model_


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
                                     

checkpoint = tf.keras.callbacks.ModelCheckpoint('imagenet_16_120_distilled_final_01.h5',
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
 					  rescale=1./255.,
                                          horizontal_flip=True)
                                          # ,fill_mode='nearest')
                                          
train_generator = train_datagen.flow(x_train,y_train,batch_size=32)


val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, rescale=1./255.)
val_generator = val_datagen.flow(x_test,
                                  y_test,
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
                    verbose=2)  


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



json_txt = model.to_json()
json_object = json.loads(json_txt)
with open('nas_distilled_client_model.json', 'w') as f:
    json.dump(json_object, f, indent=2)
model.save_weights('nas_distilled_client_model_weights.h5')
