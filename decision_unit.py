#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:49:05 2021

@author: behnam
"""

from tensorflow.python.keras.models import Model, Sequential, model_from_config, load_model
from tensorflow.python.keras.losses import categorical_crossentropy as logloss
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.python.keras.layers import Lambda, concatenate, Activation
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, ReLU, MaxPooling2D, Activation, Add, Input
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from utils import get_datasets
train_data, test_data = get_datasets('./ImageNet16')


### test image to see if it is correct


# 
x_train = np.array(train_data.data)




y_train = np.array(train_data.targets)

x_test = np.array(test_data.data)
y_test = np.array(test_data.targets)


X_val   = x_train[-6000:,:,:,:]
y_val   = y_train[-6000:]

x_train = x_train[:-6000,:,:,:]
y_train = y_train[:-6000]

y_val_ = y_val


model = load_model('imagenet_16_120_distilled_final_01.h5')
def preprocess_data(X, Y):
    """
    function that pre-processes the CIFAR10 dataset as per
    labels are one-hot encoded
    """
    # X = X
    # X = tf.keras.applications.inception_resnet_v2.preprocess_input(X)
    Y = tf.keras.utils.to_categorical(Y-1 , num_classes=120)
    return X, Y

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

X_val, y_val = preprocess_data(X_val, y_val)
traingen = ImageDataGenerator(
    rescale=1./255.,
    
    horizontal_flip=True)
                                          
train_generator = traingen.flow(x_train,y_train,shuffle=True,batch_size=32)


val_datagen = ImageDataGenerator(horizontal_flip=True, rescale=1./255.)
valid_generator = val_datagen.flow(X_val,
                                 y_val,shuffle=False,
                                 batch_size=32)


test_datagen = ImageDataGenerator(horizontal_flip=True, rescale=1./255. )
test_generator = test_datagen.flow(x_test,
                                 y_test,
                                 shuffle=False,
                                 batch_size=32)

####################################################################################################

batch_size=32
predict = []
true_label = []


for i in range(len(valid_generator)):
    inputs_batch, labels_batch = valid_generator[i]  # Retrieve batch of images and labels
    x_train = model.predict(inputs_batch)
    true_label.append(labels_batch)
    predict.append( x_train)
    
predict = np.concatenate(predict, axis=0)
true_label = np.concatenate(true_label, axis=0)

classified = np.zeros((len(true_label),1));
label = np.zeros((len(true_label),1));
for i in range(0, len(classified)):
        classified[i] = np.where(predict[i] == np.amax(predict[i])) 
        label[i] = (np.where(true_label[i] == np.amax(true_label[i])))    
        
        
from sklearn.metrics import accuracy_score
scores = accuracy_score(classified, label )
print('\n valid(test) main data accuracy: ' , scores)

i = 0
meta_data_train_label = np.zeros((predict.shape[0],1))
for i in range(0, len(predict)):
    if (classified[i] == label[i]):
        meta_data_train_label[i] = 1

meta_data_train = np.zeros((4, predict.shape[0]))

for i in range(predict.shape[0]):
    
    probs = predict[i]
    max_value = probs.max()
    probs_sorted= np.sort(probs)[::-1]
    least_confidence = probs_sorted[0] - probs_sorted[1]
    log_probs = np.log(probs)
    entropy = (probs*log_probs).sum()
    entropy
    std = np.std(probs)
    temp = [max_value , least_confidence, entropy, std]
    meta_data_train[:,i] = temp
    
    
    
true_class = []
false_class = []
for i in range(0, len(predict)):
    print(meta_data_train[2,i])
    if (meta_data_train_label[i] == 1):
        # true_class.append(meta_data_train[0,i])
        # true_class.append(meta_data_train[1,i])
        # true_class.append(meta_data_train[2,i])
        true_class.append(meta_data_train[3,i])
    else:
        # false_class.append(meta_data_train[0,i])
        # false_class.append(meta_data_train[1,i])
        # false_class.append(meta_data_train[2,i])
        false_class.append(meta_data_train[3,i])
        
kwargs = dict(alpha=0.5, bins=100)
true_class = np.array(true_class)
false_class = np.array(false_class)

from mpl_toolkits.axes_grid1 import make_axes_locatable
first_axis = plt.gca()
first_axis.hist(true_class, **kwargs, color='g', label='True Classified')
first_axis.hist(false_class, **kwargs, color='r', label='False Classified')
first_axis.set_ylim((30, 2500))
first_axis.get_xaxis().set_visible(False)
# plt.xticks('')
# first_axis.set_xlim((0.3, 1))
first_axis.spines['bottom'].set_visible(False)
# plt.gca().set(title='Meta data Distribution')
# plt.gca().set(title='Probability Distribution')
# plt.gca().set(title='List Confidence Distribution')
# plt.gca().set(title='Entropy Distribution')
plt.gca().set(title='Std Distribution')
plt.legend();
divider = make_axes_locatable(first_axis)
second_axis = divider.append_axes("bottom", size=2.4, pad = 0.0 , sharex=first_axis)
second_axis.hist(true_class, **kwargs, color='g', label='True Classified')
second_axis.hist(false_class, **kwargs, color='r', label='False Classified')
second_axis.spines['top'].set_visible(False)
second_axis.set_ylim((0, 30));

####################################################################################################
del x_train
# x_train_node_1_ = model_node_one.predict(valid_generator, verbose=1)
# x_train = model_exit_one.predict(x_train_node_1_, verbose=1)
batch_size=32
client_predict = []
true_label = []


for i in range(len(test_generator)):
    inputs_batch, labels_batch = test_generator[i]  # Retrieve batch of images and labels
    x_train = model.predict(inputs_batch)
    true_label.append(labels_batch)
    client_predict.append( x_train)
    
client_predict = np.concatenate(client_predict, axis=0)
true_label = np.concatenate(true_label, axis=0)


   
del predict
classified = np.zeros((len(true_label),1));
label = np.zeros((len(true_label),1));
for i in range(0, len(classified)):
        classified[i] = np.where(client_predict[i] == np.amax(client_predict[i])) 
        label[i] = (np.where(true_label[i] == np.amax(true_label[i])))    
        
        
from sklearn.metrics import accuracy_score
scores = accuracy_score(classified, label )
print('\n test(valid) main data accuracy: ' , scores)

meta_data_test_label = np.zeros((client_predict.shape[0],1))
for i in range(0, len(client_predict)):
    if (classified[i] == label[i]):
        meta_data_test_label[i] = 1

meta_data_test = np.zeros((4, client_predict.shape[0]))

for i in range(client_predict.shape[0]):
    
    probs = client_predict[i]
    max_value = probs.max()
    probs_sorted= np.sort(probs)[::-1]
    least_confidence = probs_sorted[0] - probs_sorted[1]
    log_probs = np.log(probs)
    entropy = (probs*log_probs).sum()
    entropy
    std = np.std(probs)
    temp = [max_value , least_confidence, entropy, std]
    meta_data_test[:,i] = temp
####################################################################################################

meta_data_train = np.transpose(meta_data_train)
meta_data_test = np.transpose(meta_data_test)


model_du = Sequential()
model_du.add(Dense(12, input_dim=4, activation='relu'))
model_du.add(Dense(8, activation='relu'))
model_du.add(Dense(1, activation='sigmoid'))
#model_.summary()

# compile the keras model
model_du.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model_du.fit(meta_data_train, meta_data_train_label, epochs=10, batch_size=10)


meta_data_test_predict = model_du.predict_classes(meta_data_test, verbose=1)

scores = accuracy_score(meta_data_test_predict,meta_data_test_label)

print('\n Decision unit accuracy: ' , scores)

du_probs = model_du.predict(meta_data_test, verbose=1)

# with open('first_exit_probs.npy', 'wb') as f:
#     np.save(f, probs) 
    
# with open('first_exit_preds.npy', 'wb') as f:
#     np.save(f, meta_data_test_predict) 
  
model_du.save('model_du.h5')  

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('model_du.h5')
tfmodel = converter.convert() 
open ('model_du.tflite' , "wb") .write(tfmodel)
print('TFLite is saved')









tr_index = 0

final_result = np.zeros((100 , 6))

threshold  = np.linspace(0,1,100)





# server_model = load_model('last_model_inception_resnet_v2.h5')
# del x_train
# server_predict = np.zeros(shape=(10000,10))
# for i, (inputs_batch, labels_batch) in enumerate(test_generator):
#     print(i)
#     if i * batch_size >= 10000:
#         break   
#     # pass the images through the network
#     x_train = server_model.predict(inputs_batch)
#     server_predict[i * batch_size : (i + 1) * batch_size] = x_train
   

# server_classified = np.zeros((len(true_label),1));
# label = np.zeros((len(true_label),1));
# for i in range(0, len(server_classified)):
#         server_classified[i] = np.where(server_predict[i] == np.amax(server_predict[i])) 
#         label[i] = (np.where(true_label[i] == np.amax(true_label[i])))    
    
# scores = accuracy_score(server_classified, label )
# print('\n server accuracy: ' , scores)

# server_decision = (server_classified == label)

server_decision = np.load('server_decision.npy')
for tr in threshold:
        server_correct_numbr = 0
        tn = 0
        tp = 0
        fn = 0
        fp = 0
     
        for i in range(0,len(du_probs)):
            label = meta_data_test_label[i]
            f_prob = du_probs[i]
            
            if (f_prob >= tr):
                exit_decision = 1
            else:
                exit_decision = 0
                
            if (exit_decision == 1 and label == 1):
                tp = tp + 1
            elif(exit_decision == 0 and label == 0):
                tn = tn + 1
            elif(exit_decision == 0 and label == 1):
                fn = fn + 1
            elif(exit_decision == 1 and label == 0):
                fp = fp + 1
                
                
            if (exit_decision == 1):
                continue
            else:
                if (server_decision[i] == 1):
                       server_correct_numbr = server_correct_numbr + 1 
                    
        
        print(tr_index)
        final_result[tr_index,:] = tr  ,tp, tn, fp, fn, server_correct_numbr
        tr_index = tr_index + 1
        
final_depict = np.zeros((100,3))
        
for i in range(len(final_result)):
    # print(i)
    whole_sending = final_result[i,2] + final_result[i,4]
    classified_sample = 6000-whole_sending
    true_server = final_result[i,5]
    # acc = final_result[i,2] + final_result[i,6] + (0.87*whole_sending)
   
    acc = final_result[i,1]  + (true_server)
    acc = acc/6000
    final_depict[i,:] = final_result[i,0], classified_sample/6000 ,acc
    
  ####################################################################################################

####################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Create some mock data

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel("Decision Unit's Sensitivity")
ax1.set_ylabel('Overall Accuracy', color=color)
ax1.plot(threshold , final_depict[:,2], color=color)
plt.grid()
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Locally Classified', color=color)  # we already handled the x-label with ax1
ax2.plot(threshold , final_depict[:,1], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('ESAI performance on ImageNet16-120', kwargs)
plt.show()  

# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np


# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Make data.
# X = first_threshold
# Y = last_threshold
# X, Y = np.meshgrid(X, Y)

# Z = np.reshape(final_depict[:,4] , (100,100))

# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(0.5, 0.95)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

# ####################################################################################################

# ####################################################################################################



# sample_index = 0
# second_threshold = 0.8
# last_threshold = 0.7

# first_exit_number = 0
# last_exit_number = 0
# import time
# first_exit_time = []
# last_exit_time = []

# for i in range(0,len(valid_generator)):
#     temp = (valid_generator[i][0])
#     for j in range(0,valid_generator[i][1].shape[0]-1):
#         start_time = time.time()
#         #print(sample_index)
# #            y_train[k]= (np.where(((test_generator[i][1][j])) == np.amax((test_generator[i][1][j]))))
# #            np.amax((test_generator[i][1][j]));
#         sample = temp[j,:,:,:]
#         sample = np.reshape(sample, [1,224,224,3])
#         node_1 = model_node_one.predict(sample)
#         exit_2 =  model_exit_one.predict(node_1)
        
    
#         max_value = exit_2.max()
#         exit_2 = exit_2[0]
#         probs_sorted= np.sort(exit_2)[::-1]
#         least_confidence = probs_sorted[0] - probs_sorted[1]
#         log_probs = np.log(exit_2)
#         entropy = (exit_2*log_probs).sum()
#         std = np.std(exit_2)
#         temp_features = [max_value , least_confidence, entropy, std]
        
#         temp_features = np.reshape(temp_features, [1,4])
        
#         du_second_value = first_exit_du.predict(temp_features)
        
#         sending_classified = meta_data_test_label[sample_index]
        

#         if (du_second_value >= second_threshold):
#             sending_decision = 1
#         else:
#             sending_decision = 0
            
        
#         if (sending_decision == 1):
#             # print("---first exit time:  %s seconds ---" % (time.time() - start_time))
#             first_exit_time.append(time.time() - start_time)
#             continue
#         else:
#             exit_last =  model_exit_last.predict(node_1)
    
#             max_value = exit_last.max()
#             exit_last = exit_last[0]
#             probs_sorted= np.sort(exit_last)[::-1]
#             least_confidence = probs_sorted[0] - probs_sorted[1]
#             log_probs = np.log(exit_last)
#             entropy = (exit_last*log_probs).sum()
#             std = np.std(exit_last)
#             temp_features = [max_value , least_confidence, entropy, std]
            
#             temp_features = np.reshape(temp_features, [1,4])
            
#             du_last_value = first_exit_du.predict(temp_features)
            
#             sending_classified = meta_data_test_label[sample_index]
#             sample_index = sample_index + 1
#             if (du_last_value >= last_threshold):
#                 sending_decision = 1
#             else:
#                 sending_decision = 0
#             if (sending_decision == 0):
#                 last_exit_number = last_exit_number +1
#             last_exit_time.append(time.time() - start_time)
# first_exit_time = np.array(first_exit_time)
# first_mean = np.mean(first_exit_time)
# last_exit_time = np.array(last_exit_time)
# last_mean = np.mean(last_exit_time)



# ####################################################################################################

# ####################################################################################################



