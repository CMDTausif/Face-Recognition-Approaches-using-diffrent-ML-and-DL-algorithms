#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow
import matplotlib.image as mpimg

data= np.load('olivetti_faces.npy')
target = np.load('olivetti_faces_target.npy')

print("Data - ",data.shape)
print("Target - ",target.shape)

# Sample images of a subject
no_of_img = 10
plt.figure(figsize=(24,24))
for i in range(no_of_img):
    plt.subplot(1,10,i+1)
    x=data[i+40] # 4th subject
    imshow(x)
plt.show()

#images of 40 distinct people
fig = plt.figure(figsize=(24,10))
cols = 10
rows = 4
for i in range(1, cols*rows+1):
    img = data[10*(i-1),:,:]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img , cmap= plt.get_cmap('gray'))
    plt.title(" Face_id: {}".format(i),fontsize = 14)
    plt.axis("off")

plt.suptitle(" Images of 40 distinct person - ", fontsize= 20)
plt.show()

#Image data in matrix from is being converted to vector
y = target.reshape(-1,1) #storing target images in Y
x = data.reshape(data.shape[0], data.shape[1] * data.shape[2]) # reshaping and storing image in X
print(" X shape - ",x.shape)
print(" Y shape - ",y.shape)
print()
#splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 1)

print(" X_train : ",x_train.shape)
print(" X_test  : ", x_test.shape)
print(" Y_train : ", y_train.shape)
print(" Y_test  : ", y_test.shape)
print()

""" spliting the dataset into training and testing set """
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

""" Training the XGBoost on the training set """
from xgboost import XGBClassifier
classifier  =  XGBClassifier()
classifier.fit(x_train, y_train)

""" Making the Confusion Matrix """
from sklearn.metrics import confusion_matrix , accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()
ac = accuracy_score(y_test , y_pred)
print("Accuracy Score- \n",ac)
print()

""" Applying K fold cross validation """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 10 test fold
accuracies = cross_val_score(estimator= classifier, X = x_train, y = y_train, cv = 10)
print(" Accuracy after K FOld cross validation XGBoost: {:.2f} %".format(accuracies.mean()*100)) # multipling it by 100 to get the value in percentage
# print(" Standard Deviation: {:.2f} %".format(accuracies.std()*100)) # multipling it by 100 to get the value in percentage


