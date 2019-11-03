# -*- coding: utf-8 -*-


#from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
#from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  Convolution2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D

#from utils.datasets import DataManager
#from utils.datasets import split_data
#from utils.preprocessor import preprocess_input
import cv2
import pandas as pd
import numpy as np


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data    



def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x



# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50


# data generator
data_generator = ImageDataGenerator(
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
#model = mini_XCEPTION(input_shape, num_classes)
#model.compile(optimizer='adam', loss='categorical_crossentropy',
#              metrics=['accuracy'])
#model.summary()


classifier=Sequential()

#classifier.add(Convolution2D(filters=32, kernel_size=(3, 3),padding='same', input_shape = (64, 64, 1), activation='relu' ))
#classifier.add(AveragePooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(filters=32, kernel_size=(5, 5),padding='same', input_shape = (64, 64, 1), activation='relu' ))
#classifier.add(BatchNormalization())
classifier.add(AveragePooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(filters=64, kernel_size=(3, 3),padding='same', input_shape = (64, 64, 1), activation='relu' ))
classifier.add(AveragePooling2D(pool_size=(2,2)))
#classifier.add(Convolution2D(filters=128, kernel_size=(3, 3),padding='same', input_shape = (64, 64, 1), activation='relu' ))

classifier.add(AveragePooling2D(pool_size=(2,2)))
classifier.add(Flatten())


classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=7,activation='softmax',name='prediction'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




datasets = ['fer2013']
dataset_name='fer2013'
print('Training dataset:', dataset_name)

# callbacks
#log_file_path = dataset_name + '_emotion_training.log'
#csv_logger = CSVLogger(log_file_path, append=False)
#early_stop = EarlyStopping('val_loss', patience=patience)
#reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
#                              patience=int(patience/4), verbose=1)
trained_models_path = dataset_name + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
#model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
 #                                               save_best_only=True)
#callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# loading dataset
#data_loader = DataManager(dataset_name, image_size=input_shape[:2])
#faces, emotions = data_loader.get_data()

data = pd.read_csv('fer2013.csv')
pixels = data['pixels'].tolist()
width, height = 48, 48
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    face = cv2.resize(face.astype('uint8'), input_shape[:2])
    faces.append(face.astype('float32'))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
emotions = pd.get_dummies(data['emotion']).as_matrix()

faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
train_data, val_data = split_data(faces, emotions, validation_split)


num_samples = len(faces)
num_train_samples = int((1 - validation_split)*num_samples)
train_x = faces[:num_train_samples]
train_y = emotions[:num_train_samples]
val_x = faces[num_train_samples:]
val_y = emotions[num_train_samples:]
train_data = (train_x, train_y)
val_data = (val_x, val_y)


train_faces, train_emotions = train_data
#model.fit_generator(data_generator.flow(train_faces, train_emotions,
classifier.fit_generator(data_generator.flow(train_faces, train_emotions,
                                   batch_size),
                    steps_per_epoch=len(train_faces) / batch_size,
                    epochs=num_epochs, verbose=1, 
                    validation_data=val_data)
    
    #lassifier.fit_generator(train_data,
