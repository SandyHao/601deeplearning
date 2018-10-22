#source：CSDN 
#copyright：https://blog.csdn.net/pengdali/article/details/79050662 

# -*- coding: utf-8 -*-
import os
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,Flatten,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
 
class PowerTransferMode:
    #data generator; it will cause trouble to computer if load all the images into ram at the same time
    def DataGen(self, dir_path, img_row, img_col, batch_size, is_train):
        if is_train:
            datagen = ImageDataGenerator(rescale=1./255,
                zoom_range=0.25, rotation_range=15.,
                channel_shift_range=25., width_shift_range=0.02, height_shift_range=0.02,
                horizontal_flip=True, fill_mode='constant')
        else:
            datagen = ImageDataGenerator(rescale=1./255)
 
        generator = datagen.flow_from_directory(
            dir_path, target_size=(img_row, img_col),
            batch_size=batch_size,
            #class_mode='binary',
            shuffle=is_train)
 
        return generator
 
    #VGG model
    def VGG19_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True, is_plot_model=False):
        color = 3 if RGB else 1
        base_model = VGG19(weights='imagenet', include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
                              classes=nb_classes)
 
        # fix all the layers of base_model，get correct bottleneck feature
        for layer in base_model.layers:
            layer.trainable = False
 
        x = base_model.output
        #add own full connection layer
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
 
        #train model
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
        #plot model
        if is_plot_model:
            plot_model(model, to_file='vgg19_model.png',show_shapes=True)
 
        return model
 
    
    #train model
    def train_model(self, model, epochs, train_generator, steps_per_epoch, validation_generator, validation_steps, model_url, is_load_model=False):
        #load model
        if is_load_model and os.path.exists(model_url):
            model = load_model(model_url)
 
        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)
        # save model
        model.save(model_url,overwrite=True)
        return history_ft
 
    # plot loss
    def plot_training(self, history):
      acc = history.history['acc']
      val_acc = history.history['val_acc']
      loss = history.history['loss']
      val_loss = history.history['val_loss']
      epochs = range(len(acc))
      plt.plot(epochs, acc, 'b-')
      plt.plot(epochs, val_acc, 'r')
      plt.title('Training and validation accuracy')
      plt.figure()
      plt.plot(epochs, loss, 'b-')
      plt.plot(epochs, val_loss, 'r-')
      plt.title('Training and validation loss')
      plt.show()
 
 
if __name__ == '__main__':
    image_size = 197
    batch_size = 32
 
    transfer = PowerTransferMode()
 
    #load data
    train_generator = transfer.DataGen('./data2/train', image_size, image_size, batch_size, True)
    validation_generator = transfer.DataGen('./data2/validation', image_size, image_size, batch_size, False)
 
    VGG19
    model = transfer.VGG19_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    history_ft = transfer.train_model(model, 10, train_generator, 60, validation_generator, 60, 'vgg19_model_weights.h5', is_load_model=False)
 
   
    #plot acc_loss
    transfer.plot_training(history_ft)
