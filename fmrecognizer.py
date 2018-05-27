#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 07:14:46 2018

@author: kshitij
"""




from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical


# Second dimension of the feature is dim2
feature_dim_2 = 11

# Save data to array file first
save_coefficients(max_len=feature_dim_2)

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_split_data_train_test()

# # Feature dimension
feature_dim_1 = 20
channel = 1
epochs = 115
batch_size = 10
verbose = 1
num_classes = 1


# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)


y_train_hot = to_categorical(y_train, 0)
y_test_hot = to_categorical(y_test, 0)


#relu - to get non linear output
def get_model():
    model = Sequential()
    #convolution
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', 
                     input_shape=(feature_dim_1, feature_dim_2, channel)))
    #pooling - reduces the complexity
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Adding one more conv layer for making a deeper network
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))
    #Flattening - Takes all the pooled feature maps and put into
    #a single 1-dim array
    model.add(Flatten())
    #Full connection
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    #Compiling
    model.compile(optimizer='adam',loss = 'binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = get_model()
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test))

# Predicts one sample
def predict(filepath, model):
    sample = to_mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    result = model.predict(sample_reshaped)    
    if result > 0.5:
        prediction = 'Song track'
    else:
        prediction = 'Advertisement'
    return prediction

print(predict('./predict/ad101.wav', model=model))

print(predict('./predict/ad102.wav', model=model))
print(predict('./predict/ad103.wav', model=model))
print(predict('./predict/ad104.wav', model=model))
print(predict('./predict/ad105.wav', model=model))
print(predict('./predict/ad106.wav', model=model))
print(predict('./predict/ad107.wav', model=model))
print(predict('./predict/ad108.wav', model=model))
print(predict('./predict/ad109.wav', model=model))

print('************')
print(predict('./predict/track101.wav', model=model))

print(predict('./predict/track102.wav', model=model))
print(predict('./predict/track103.wav', model=model))
print(predict('./predict/track104.wav', model=model))
print(predict('./predict/track105.wav', model=model))
print(predict('./predict/track106.wav', model=model))
print(predict('./predict/track107.wav', model=model))
print(predict('./predict/track108.wav', model=model))
print(predict('./predict/track109.wav', model=model))
print(predict('./predict/track110.wav', model=model))
"""