import mysql.connector as conn
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
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import keras.backend as K

train_data = 'dme_data/train'
test_data = 'dme_data/test'

def one_hot_label(img):
    label = img.split('-')[0]
    if label == 'NORMAL':
        ohl = np.array([1,0])
    elif label == 'DME':
        ohl = np.array([0,1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(128,128))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(128,128))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return test_images


training_images = train_data_with_label()
testing_images = test_data_with_label()
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1) 
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

"""mySQLconnection = conn.connect(host='localhost',
                             database='models',
                             user='root',
                             password='gkhan.*1181?')"""
"""sql_select_Query = "select * from modelss"
cursor = mySQLconnection .cursor()
cursor.execute(sql_select_Query)
records = cursor.fetchall()"""
"""f= open("conv_sizes.txt","a")


for t in records:
    f.write(' '.join(str(s) for s in t) + '\n')
f.close()"""



"""f = open('conv_sizes.txt', 'r')
x = f.readlines()
f.close()
a = []
for i in x:
    a.append((int(x[0].split()[]))
print(int(x[0].split()[1]))"""
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
a = []
#save = save_database()
def conv_3f(pool):
    if pool % 2 == 0:
        for j in single:
            conv_3 = j
            model2.add(Conv2D(filters=3,kernel_size=j,strides=1,activation='relu'))
            model2.add(MaxPool2D(pool_size=2))
            model2.summary()
            last_layer = model2.layers[5].output.shape[1]
            val = (conv_1,conv_2,conv_3,int(last_layer))
            a.append(val)
            #save.save_models(val)
            model2.pop()
            model2.pop()      
    else:
        for j in double:
            conv_3 = j
            model2.add(Conv2D(filters=3,kernel_size=j,strides=1,activation='relu'))
            model2.add(MaxPool2D(pool_size=2))
            model2.summary()
            last_layer = model2.layers[5].output.shape[1]
            val = (conv_1,conv_2,conv_3,int(last_layer))
            a.append(val)
            #save.save_models(val)
            model2.pop()
            model2.pop() 

for t in content:
    conv_1 = t
    if conv_1 % 2 == 0 :
        model2 = Sequential()
        model2.add(Conv2D(filters=3,kernel_size=t,strides=1,activation='relu',input_shape=(227, 227, 3)))
        model2.add(MaxPool2D(pool_size=2))
        out_1 = model2.output_shape[1]
        if out_1 % 2 ==0:
  
            for i in single:
                model2.add(Conv2D(filters=3,kernel_size=i,strides=1,activation='relu'))
                model2.add(MaxPool2D(pool_size=2))
                pool = model2.output_shape[1]
                conv_2 = i
                conv_3f(pool)
                model2.pop()
                model2.pop()

            model2.pop()
            model2.pop()

        else:
 
            for i in double:
                model2.add(Conv2D(filters=3,kernel_size=i,strides=1,activation='relu'))
                model2.add(MaxPool2D(pool_size=2))
                pool = model2.output_shape[1]
                conv_2 = i
                conv_3f(pool)
                model2.pop()
                model2.pop()

            model2.pop()
            model2.pop()
"""fname = "kernel_size.txt"
single = []
double = []

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
conv_1 = 0
conv_2 = 0
conv_3 = 0
conv_4 = 0
#print(content)

#save = save_database()
a = []
def conv_last(last_layer,conv_3):
    if last_layer % 2 ==0:
        for r in single:
            conv_4 = r
            model.add(Conv2D(filters=3,kernel_size=r,strides=1,activation='relu'))
            model.add(MaxPool2D(pool_size=2))
            model.summary()
            last_layer = model.layers[7].output.shape[1]
            model.pop()
            model.pop()
            val = (conv_1,conv_2,conv_3,conv_4,int(last_layer))
            #save.save_models(val)
            a.append(val)
    else:
        for r in double:
            conv_4 = r
            model.add(Conv2D(filters=3,kernel_size=r,strides=1,activation='relu'))
            model.add(MaxPool2D(pool_size=2))
            model.summary()
            last_layer = model.layers[7].output.shape[1]
            model.pop()
            model.pop()
            val = (conv_1,conv_2,conv_3,conv_4,int(last_layer))
            #save.save_models(val)
            a.append(val)

def conv__3(pool):
    if pool % 2 == 0:
        for j in single:
            conv_3 = j
            model.add(Conv2D(filters=3,kernel_size=j,strides=1,activation='relu'))
            model.add(MaxPool2D(pool_size=2))
            #model.summary()
            last_layer = model.layers[5].output.shape[1]
            ##################
            conv_last(last_layer,conv_3)
            ##################
            model.pop()
            model.pop()      
    else:
        for j in double:
            conv_3 = j
            model.add(Conv2D(filters=3,kernel_size=j,strides=1,activation='relu'))
            model.add(MaxPool2D(pool_size=2))
            #model.summary()
            last_layer = model.layers[5].output.shape[1]
            conv_last(last_layer,conv_3)
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
                conv__3(pool)
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
                conv__3(pool)
                model.pop()
                model.pop()

            model.pop()
            model.pop()"""

K.clear_session()
mst_acc = 0
sayac=0
#b = a[110:]
for row in a:
    for p in range(5):
        sayac= sayac+1

        model = Sequential()

        model.add(InputLayer(input_shape=[128,128,1]))
        model.add(Conv2D(filters=32,kernel_size=row[0],strides=1,activation='relu'))
        #model.add(Conv2D(filters=32,kernel_size=row[0],strides=1,activation='relu'))
        model.add(MaxPool2D(pool_size=2,strides=(2, 2)))

        model.add(Conv2D(filters=64,kernel_size=row[1],strides=1,activation='relu'))
        #model.add(Conv2D(filters=64,kernel_size=row[1],strides=1,activation='relu'))
        model.add(MaxPool2D(pool_size=2,strides=(2, 2)))


        model.add(Conv2D(filters=128,kernel_size=row[2],strides=1,activation='relu'))
        #model.add(Conv2D(filters=128,kernel_size=row[2],strides=1,activation='relu'))
        model.add(MaxPool2D(pool_size=2,strides=(2, 2)))
        
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(row[3]*row[3],activation='relu'))
        model.add(Dense(row[3]*row[3],activation='relu'))
        model.add(Dense(2,activation='softmax'))
        Optimizer = Adam(lr=0.75e-5)

        #model.compile(optimizer=Optimizer,loss='categorical_crossentropy',metrics=['accuracy']) #Parametrelerine bak
        #model.fit(x=tr_img_data,y=tr_lbl_data,epochs=75,batch_size=100) #Cross entropy grafiği MAE MSE
        #checkpoint = keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')  

        model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(x=tr_img_data,y=tr_lbl_data,batch_size=10, epochs=250, verbose=1)
        model.summary()


        #model.save_weights("modelDME.h5")
        dme = 0
        normal = 0
        #fig = plt.figure(figsize=(10,10))
        y_pred = []
        y_test = []
        for cnt, data in enumerate(testing_images[:]):
            img = data[0]
            y_test.append(data[1][1])
            data = img.reshape(1,128,128,1)
            model_out = model.predict([data])
            y_pred.append(np.argmax(model_out))
            if np.argmax(model_out) == 1:
                str_label = 'DME'
            else:
                str_label = 'NORMAL'
            #plt.imshow(img)
            #plt.title(str_label)
            #plt.show()
            #print(np.argmax(model_out))

        from sklearn.metrics import confusion_matrix #y_test / y_pred
        import seaborn as sns

        confusion_mtx = confusion_matrix(y_test, y_pred)
        TN = confusion_mtx[0][0]
        TP = confusion_mtx[1][1]
        FP = confusion_mtx[0][1]
        FN = confusion_mtx[1][0]
        S = TN+TP+FP+FN

        accuracy = (TP+TN)/S
        recall_sensitivity = (TP/(TP+FN))
        specificity = (TN/(TN+FP))
        f1score = ((2*TP)/((2*TP)+FP+FN))
        true_positive_rate = (TP/(TP+FN)) #sensitivity
        false_positive_rate = (FP/(TN+FP)) #1-specificity


        print("Accuracy : ",accuracy)
        print("Recall Sensitivity : ",recall_sensitivity)
        print("Specificity : ",specificity)
        print("F1 Score : ",f1score)
        print("True positive rate : ",true_positive_rate)
        print("False negative rate : ",false_positive_rate)

        if mst_acc < accuracy:
            mst_acc = accuracy
            model.save_weights("modelDME.h5")
        f= open("trainme.txt","a+")
        strr = str(row[0])+"-"+str(row[1])+"-"+str(row[2])
        """f.write(str(sayac))
        f.write("-")
        """
        f.write(strr)
        f.write(":")
        f.write(str(accuracy)+":"+str(recall_sensitivity)+":"+str(specificity)+":"+str(f1score)+":"+str(true_positive_rate)+":"+str(false_positive_rate))

        f.write('\n')
        f.close()
        #sql = "INSERT INTO multi_train (model_id,acc) VALUES(%s,%s)"
        try:
            if sayac % 250 == 0:
                gauth = GoogleAuth()
                gauth.LocalWebserverAuth()
                drive = GoogleDrive(gauth)
                file1 = drive.CreateFile()
                file1.SetContentFile("trainme.txt")
                file1.Upload()
        except:
            continue
        
        K.clear_session()

        #cursor.execute(sql,arr)  
        #mySQLconnection.commit()
        
        """
        sns.heatmap(confusion_mtx, annot=True, fmt="d");
        plt.show()"""

        """model_json = model.to_json()
        with open("modelDME.json", "w") as json_file:
            json_file.write(model_json)

        """