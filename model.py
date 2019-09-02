import csv
import cv2
import os
import numpy as np
import datetime
import random
from math import ceil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda

samples = []
batch_size=128

#load csv file
with open('./train_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
#resize image and convert bgr to rgb
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[60:130, :]
    image = cv2.resize(image, (160,70))
    return image

#Splitting training and validation sets
training_data, valid_data = train_test_split(samples, test_size = 0.2)

#generator
def generator(data, batchSize = 128):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), batch_size):
            X_batch = []
            y_batch = []
            images = []
            angles = []
            details = data[i: i+batch_size]
            for line in details:
                center_image = preprocess(cv2.imread('./train_data/IMG/'+ line[0].split('/')[-1]))
                steering_angle = float(line[3])
                # original image
                images.append(center_image)
                angles.append(steering_angle)
                # flipped image
                images.append(np.fliplr(center_image))
                angles.append(-steering_angle)
                # left camera image and steering angle with offset
                images.append(preprocess(cv2.imread('./train_data/IMG/'+ line[1].split('/')[-1])))      
                angles.append(steering_angle+0.2)
                # right camera image and steering angle with offset
                images.append(preprocess(cv2.imread('./train_data/IMG/'+ line[2].split('/')[-1])))  
                angles.append(steering_angle-0.2)
            # converting to numpy array
            X_batch = np.array(images)
            y_batch = np.array(angles)
            yield shuffle(X_batch, y_batch)

#model

model = Sequential()
model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

#compiling and running the model
train_generator = generator(training_data, batch_size)
validation_generator = generator(valid_data, batch_size)
model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(training_data)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(valid_data)/batch_size),
            epochs=10, verbose=1)
#saving the model
model.save('model.h5')