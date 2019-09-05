import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

import numpy as np
import os
import wandb
from wandb.keras import WandbCallback

wandb.init()

config = wandb.config
config.batch_size = 128
config.epochs = 10
config.learn_rate = 0.001

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#comment

model = Sequential() 
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(num_classes, activation="softmax")) 
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(config.learn_rate),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test), 
    callbacks=[WandbCallback(data_type="image", labels=class_names)])
