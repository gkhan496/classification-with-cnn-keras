import time
import cv2
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard
from save_models import save_database


fname = "kernel_size.txt"
single = []
double = []

conv_1 = 0
conv_2 = 0
conv_3 = 0

with open(fname) as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for i in range(len(content)):
    content[i] = int(content[i])

for i in content:
    if i % 2 ==0:
        double.append(i)

    else:
        single.append(i)

save = save_database()
def conv_3f(pool):
    if pool % 2 == 0:
        for j in single:
            conv_3 = j
            model.add(Conv2D(filters=3,kernel_size=j,strides=1,activation='relu'))
            model.add(MaxPool2D(pool_size=2))
            model.summary()
            last_layer = model.layers[5].output.shape[1]
            val = (conv_1,conv_2,conv_3,int(last_layer))
            save.save_models(val)
            model.pop()
            model.pop()      
    else:
        for j in double:
            conv_3 = j
            model.add(Conv2D(filters=3,kernel_size=j,strides=1,activation='relu'))
            model.add(MaxPool2D(pool_size=2))
            model.summary()
            last_layer = model.layers[5].output.shape[1]
            val = (conv_1,conv_2,conv_3,int(last_layer))
            save.save_models(val)
            model.pop()
            model.pop() 

for t in content:
    conv_1 = t
    if conv_1 % 2 == 0 :
        model = Sequential()
        model.add(Conv2D(filters=3,kernel_size=t,strides=1,activation='relu',input_shape=(227, 227, 3)))
        model.add(MaxPool2D(pool_size=2))
        out_1 = model.output_shape[1]
        if out_1 % 2 ==0:
  
            for i in single:
                model.add(Conv2D(filters=3,kernel_size=i,strides=1,activation='relu'))
                model.add(MaxPool2D(pool_size=2))
                pool = model.output_shape[1]
                conv_2 = i
                conv_3f(pool)
                model.pop()
                model.pop()

            model.pop()
            model.pop()

        else:
 
            for i in double:
                model.add(Conv2D(filters=3,kernel_size=i,strides=1,activation='relu'))
                model.add(MaxPool2D(pool_size=2))
                pool = model.output_shape[1]
                conv_2 = i
                conv_3f(pool)
                model.pop()
                model.pop()

            model.pop()
            model.pop()