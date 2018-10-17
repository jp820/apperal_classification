import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### --- MODEL definition --- ###
img_width, img_height = 300, 300

model = Sequential()

### Conv layer 1 ###
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

### Conv layer 2 with dropout ###
model.add(Conv2D(64, (3, 3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))

### Conv layer 3 ###
model.add(Conv2D(64, (3, 3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

### Conv layer 4 with dropout ###
model.add(Conv2D(128, (3, 3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))

### Flatten 3d features to 1d vectors using flatten
model.add(Flatten()) 

### FC layer ###
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))

### Final layer with softmax ###
model.add(Dense(13))
model.add(BatchNormalization())
model.add(Activation('softmax'))

### Compile the model with categorical crossentrpy loss and adam optimizer ###
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


### --- Data pipeline --- ###

### Image dimensions for network ###
img_width, img_height = 300, 300

### Directory paths ###
train_path = 'training_data'
validation_path = 'validation_data'
batch_size = 8

### Build train data generator ###
train_datagen = ImageDataGenerator(
        rescale=1./255
        )

### Using data generator, generate batches for training iterations ###
train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(img_width, img_height),  # all images will be resized to 300x300
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

### Build Validation data generator ###
test_datagen = ImageDataGenerator(rescale=1./255)

### Use validation generator for validation data batches ###
validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

### --- Training ---###
SOURCE_FOLDER = '/content/drive/My Drive/Asgn1'
history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=320 // batch_size)

### Save trainined model ###
SOURCE_FOLDER = '/content/drive/My Drive/Asgn1'
model.save(os.path.join(SOURCE_FOLDER ,'config5_100_epochs_try1.h5'))

### --- Training summary plots --- ###

### Summarize history for accuracy ###
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

### Summarize history for loss ###
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


