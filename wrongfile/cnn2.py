import os  # 处理字符串路径
import glob  # 查找文件
from keras.models import Sequential  # 导入Sequential模型
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from pandas import Series, DataFrame
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
#加载数据
import os
from PIL import Image
import numpy as np
#读取文件夹train下的1000张图片，图片为彩色图，rgb*3
#如果是将彩色图作为输入,图像大小224*224
def load_data():
    #sed = 1000
    #data = np.empty((2000,224,224,3),dtype="float32")
    #label = np.empty((2000,))
    path = './train'  
    files = os.listdir(path)  
    images = []  
    labels = []  
    num = 4000  
    t = 0  
    for f in files:  
        img_path = path + '/' + f  
        if t >= num / 2 and 'cat' in f:  
           continue  
        if t == num:  
            break  
        from keras.preprocessing import image  
        img = image.load_img(img_path, target_size=(224, 224))  
        img_array = image.img_to_array(img)  
        images.append(img_array)  
        if 'cat' in f:  
            labels.append(0)  
        else:  
            labels.append(1)  
        t += 1  
      
    data = np.array(images)  
    labels = np.array(labels)  
      
    label = np_utils.to_categorical(labels,2)  #label = np_utils.to_categorical(labels,2)
    return data, label  

data,label = load_data()
print(data.shape)
train_data = data[:3600]
train_labels = label[:3600]
validation_data = data[3600:]
validation_labels = label[3600:]


model = Sequential()
#第一个卷积层，4个卷积核，每个卷积核大小5*5。
#激活函数用tanh
#你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
model.add(Convolution2D(4, 5, 5,input_shape=(224, 224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#第二个卷积层，8个卷积核，每个卷积核大小3*3。
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#第三个卷积层，16个卷积核，每个卷积核大小3*3
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#全连接层，先将前一层输出的二维特征图flatten为一维的。

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#sigmoid分类，输出是2类别
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(train_data, train_labels,
         nb_epoch=20, batch_size=200,
         validation_data=(validation_data, validation_labels))
'''
from keras.utils import plot_model
plot_model(model, to_file='./model.png')
'''