import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Flatten, Dense

import numpy as np
import os
import cv2
import wandb
from wandb.keras import WandbCallback

train_test_ratio = 0.8
img_width, img_height = 150, 200
epochs = 5
batch_size = 64

wandb.init(project = "genralize")
wandb.config.train_test_ratio = train_test_ratio
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size


#load, resize and convert from color to grayscale
def load_images_resize_bw(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/1.0
            img = cv2.resize(img, dsize=(150, 200), interpolation=cv2.INTER_AREA)
            img = np.array(img)
            images.append(img)
    return images


notes = load_images_resize_bw('C:/Users/sushant/Desktop/iv_resources/github/notes')
memes = load_images_resize_bw('C:/Users/sushant/Desktop/iv_resources/github/memes')

notes = np.array(notes)
memes = np.array(memes)
notes_labels = np.ones(notes.shape[0])
memes_labels = np.zeros(memes.shape[0])

X = np.concatenate((notes,memes))
Y = np.concatenate((notes_labels,memes_labels))

np.random.seed(0)
np.random.shuffle(X)
np.random.seed(0)
np.random.shuffle(Y)



nb_train_samples = int(train_test_ratio*len(X))
nb_test_samples = len(X) - int(train_test_ratio*len(X))
x_train = X[:nb_train_samples]
x_test = X[nb_test_samples:]
x_train = x_train.reshape((x_train.shape[0], img_height, img_width, 1))
x_test = x_test.reshape((x_test.shape[0], img_height, img_width, 1))
y_train = Y[:nb_train_samples]
y_test = Y[nb_test_samples:]



datagen = ImageDataGenerator(
    rescale = 1.0/255,
    zoom_range = 0.2,
    rotation_range=25,
    horizontal_flip = True)

train_data_generator = datagen.flow(
    x_train,
    y_train,
    batch_size = batch_size)

test_data_generator = datagen.flow(
    x_test,
    y_test,
    batch_size = batch_size)



model = Sequential()
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])



model.fit(
    train_data_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = test_data_generator,
    validation_steps = nb_test_samples // batch_size, callbacks=[WandbCallback()])

model.save("keras_model")