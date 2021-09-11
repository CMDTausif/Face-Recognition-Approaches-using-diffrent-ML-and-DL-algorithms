#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

""" Step2 - Loading the Dataset """
#loading the dataset
data_set = np.load("ORL_faces.npz")

#loading the "Train" images
x_train = data_set['trainX']
#normalizing every image
x_train = np.array(x_train ,dtype="float32")/255

x_test = data_set['testX']
x_test = np.array(x_test, dtype="float32")/255

#loading the label of images
y_train = data_set['trainY']
y_test = data_set['testY']

#showing the train and test images in DataFormat
print("X_train: {}".format(x_train[:]))
print()
print("Y_train shape: {}".format(y_train))
print()
print("X_test shape: {}".format(x_test.shape))
print()

""" step3- Splitting the dataset """
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,random_state= 0,test_size= 110)

""" Step4 - Resizing the images """
im_rows = 112
im_cols = 92
batch_size = 512
im_shape = (im_rows, im_cols, 1)

#changing the size of images
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_valid = x_valid.reshape(x_valid.shape[0], *im_shape)
print("y_train shape: {}".format(y_train.shape[0]))
print("y_test shape: {}".format(y_test.shape))
print()

""" Building the CNN model """
from keras.preprocessing.image import ImageDataGenerator
#initializing the CNN
cnn = tf.keras.models.Sequential()
#adding the 1st Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=36, kernel_size=7, activation='relu', input_shape= im_shape))
#adding 1st pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
#adding 1st convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=54, kernel_size=5, activation='relu', input_shape= im_shape))
#adding 2nd max pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
# Falttening
cnn.add(tf.keras.layers.Flatten())
#1st fully connected layer
cnn.add(tf.keras.layers.Dense(2024, activation='relu'))
#adding 1st dropout layer
cnn.add(tf.keras.layers.Dropout(0.5))
#2nd fully connected layer
cnn.add(tf.keras.layers.Dense(1024, activation='relu'))
#2nd dropout layer
cnn.add(tf.keras.layers.Dropout(0.5))
#3rd fully connected layer
cnn.add(tf.keras.layers.Dense(512, activation='relu'))
#3rd dropout layer
cnn.add(tf.keras.layers.Dropout(0.5))
#20 is the number of outputs
#final full connected layer
cnn.add(tf.keras.layers.Dense(20, activation='softmax'))

#Compiling the CNN
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

""" Step 6 - Training the model """
history = cnn.fit( np.array(x_train), np.array(y_train), batch_size=512,
    epochs=250, verbose=2, validation_data=(np.array(x_valid),np.array(y_valid)))

""" evaluating the score """

score = cnn.evaluate(np.array(x_test), np.array(y_test), verbose= 0)
print('test los {:.4f}'.format(score[0]))
print('test accuracy {:.4f}'.format(score[1]))
print()

""" Plotting the result """
#list all the data in history
import itertools

print(history.history.keys())
print()

#summarizing history for accuracy
# summarizing history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarizing history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

""" Predicting the accuracy """
from sklearn.metrics import confusion_matrix, accuracy_score
predicted_result = np.array(cnn.predict(x_test))
y_pred = cnn.predict_classes(x_test)
Accuracy = accuracy_score(y_test, y_pred)
print("accuracy : ", Accuracy)
print()
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()












