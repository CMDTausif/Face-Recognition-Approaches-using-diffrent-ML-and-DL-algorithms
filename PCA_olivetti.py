print(" PCA_OLivetti ".center(125,"*"))

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

from sklearn.decomposition import PCA
pca = PCA(100)
X_train_pca = pca.fit_transform(x_train)
X_test_pca = pca.transform(x_test)

print('Original dataset:',x_train.shape)
print('Dataset after applying PCA:',X_train_pca.shape)
print('No of PCs/Eigen Faces:',len(pca.components_))
print('Eigen Face Dimension:',pca.components_.shape)
print('Variance Captured:',np.sum(pca.explained_variance_ratio_))
print()

# Average face of the samples
plt.subplots(1,1,figsize=(8,8))
plt.imshow(pca.mean_.reshape((64,64)), cmap="gray")
plt.title('Average Face')
plt.show()

#showing all eigen faces
number_of_eigenfaces=len(pca.components_)
eigen_faces=pca.components_.reshape((number_of_eigenfaces, data.shape[1], data.shape[2]))

columns=10
rows=int(number_of_eigenfaces/columns)
fig, axarr=plt.subplots(nrows=rows, ncols=columns, figsize=(24,24))
axarr=axarr.flatten()
for i in range(number_of_eigenfaces):
    axarr[i].imshow(eigen_faces[i],cmap="gray")

    axarr[i].set_title("eigen_id:{}".format(i))
plt.suptitle("All Eigen Faces".format(10*"=", 10*"="),fontsize = 8)
plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

""" Training the Logistic Regression Model on Training set"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(X_train_pca,y_train)

""" making the Confusion matrix """
from sklearn.metrics import confusion_matrix , accuracy_score
y_pred = classifier.predict(X_test_pca)
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()

ac = accuracy_score(y_test , y_pred)
print("Accuracy Score- \n",ac)
print()

""" Applying K fold cross validation for Logistic Regression model """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 5 test fold
LR_accuracies = cross_val_score(estimator= classifier, X = X_train_pca, y = y_train, cv = 5)
print(" Accuracy in Logistic Regression after Kfold cross validation: {:.2f} %".format(LR_accuracies.mean()*100)) # multipling


