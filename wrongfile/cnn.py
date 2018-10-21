#作者：shizhengxin123 
#来源：CSDN 
#原文：https://blog.csdn.net/shizhengxin123/article/details/72473245 
#版权声明：本文为博主原创文章，转载请附上博文链接！
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
model = Sequential()
#第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
#激活函数用relu
#你还可以在model.add(Activation('relu'))后加上dropout的技巧，防止过拟合
model.add(Dropout(0.5))
model.add(Convolution2D(4, 5, 5,input_shape=(224, 224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
#激活函数用relu
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#第三个卷积层，16个卷积核，每个卷积核大小3*3
#激活函数用relu
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#全连接层，先将前一层输出的二维特征图flatten为一维的。
#全连接有128个神经元节点,初始化方式为normal
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

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './test1',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
'''
model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)
#error: you must complile your model before using it
'''
model.fit(train_data, train_labels, nb_epoch=50,batch_size=100,
         validation_data=(validation_data,validation_labels))
from keras.utils import plot_model
plot_model(model, to_file='./model.png')

