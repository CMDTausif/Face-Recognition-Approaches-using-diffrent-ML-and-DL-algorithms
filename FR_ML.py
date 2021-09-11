print(" Face Recognition Using Diff ML Algorithm ".center(125,"*"))
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

#storing accuracies of ML algorithms for comapring those accuracies
list_name = []
list_accuracy = []

''' Applying 1-NN algorithm '''

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train,"\n")
print(x_test,"\n")

''' training the KNN - 1NN model on training dataset '''
from sklearn.neighbors import KNeighborsClassifier
knnclassifier  = KNeighborsClassifier( n_neighbors = 1, metric= "minkowski", p=2 )
knnclassifier.fit(x_train , y_train)

#predicting the test set
y_pred = knnclassifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print()


#making a confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()
ac = accuracy_score(y_test , y_pred)
print("Accuracy Score in 1NN- \n",ac)
print()

""" Applying K fold cross validation for KNN model """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 5 test fold
KNN_accuracies = cross_val_score(estimator= knnclassifier, X = x_train, y = y_train, cv = 5)
print(" Accuracy in 1NN after Kfold cross validation: {:.2f} %".format(KNN_accuracies.mean()*100)) # multipling it by 100 to get the value in percentage
# print(" Standard Deviation: {:.2f} %".format(accuracies.std()*100)) # multipling it by 100 to get the value in percentage

list_name.append(" 1NN Algorithm")
list_accuracy.append(KNN_accuracies)


""" Training the Logistic Regression Model on Training set"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(x_train,y_train)

""" making the Confusion matrix """
from sklearn.metrics import confusion_matrix , accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()

ac = accuracy_score(y_test , y_pred)
print("Accuracy Score in Logistic regression- \n",ac)
print()

""" Applying K fold cross validation for Logistic Regression model """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 5 test fold
LR_accuracies = cross_val_score(estimator= classifier, X = x_train, y = y_train, cv = 5)
print(" Accuracy in Logistic Regression after Kfold cross validation: {:.2f} %".format(LR_accuracies.mean()*100)) # multipling it by 100 to get the value in percentage
# print(" Standard Deviation: {:.2f} %".format(accuracies.std()*100)) # multipling it by 100 to get the value in percentage

list_name.append(" Logistic Regression Algorithm")
list_accuracy.append(LR_accuracies)


''' Training the SVM model on the training dataset '''
from sklearn.svm import SVC
SVMclassifier = SVC(kernel = "linear" , random_state= 0)
SVMclassifier.fit(x_train,y_train)

#predicting the test set
y_pred = SVMclassifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print()

#making a confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()

ac = accuracy_score(y_test , y_pred)
print("Accuracy Score in SVM- \n",ac)
print()

""" Applying K fold cross validation for SVM model """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 5 test fold
SVM_accuracies = cross_val_score(estimator= SVMclassifier, X = x_train, y = y_train, cv = 5)
print(" Accuracy in SVM after 5 fold cross validation: {:.2f} %".format(SVM_accuracies.mean()*100))

list_name.append(" SVM Algorithm")
list_accuracy.append(SVM_accuracies)

''' Training the Kernel SVM model on the training dataset -> kernel = rbf'''
from sklearn.svm import SVC
SVMclassifier = SVC(kernel = "rbf" , random_state= 0)
SVMclassifier.fit(x_train,y_train)

#predicting the test set
y_pred = SVMclassifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print()

#making a confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()

ac = accuracy_score(y_test , y_pred)
print("Accuracy Score in Kernel SVM- \n",ac)
print()

""" Applying K fold cross validation for Kernel SVM model """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 5 test fold
KSVM_accuracies = cross_val_score(estimator= SVMclassifier, X = x_train, y = y_train, cv = 5)
print(" Accuracy in Kernel SVM after 5 fold cross validation: {:.2f} %".format(KSVM_accuracies.mean()*100))

list_name.append(" Kernel SVM Algorithm")
list_accuracy.append(KSVM_accuracies)

#Accuracies of every ML algorithm is shown in list
# import pandas as pd
# df = pd.DataFrame({'METHOD': list_name, 'ACCURACY (%)': list_accuracy})
# # df = df.sort_values(by= ['ACCURACY (%)'])
# df = df.reset_index(drop=True)
# df.head()
# print(df)
print()

''' Applying Random Forest Classification '''
#Training Random Forest Classification in Training model
from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(n_estimators= 100 , criterion= 'entropy' , random_state= 0 )
RFclassifier.fit(x_train , y_train)

# #predicting a new result
# print(RFclassifier.predict(sc.transform([[30,87000]])))
# print()

#predicting Test set
y_pred = RFclassifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print()

#making a confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix for Random Forest- \n",cm)
print()
ac = accuracy_score(y_test , y_pred)
print("Accuracy Score for Random Forest- \n",ac)
print()

""" Applying K fold cross validation in Random Forest """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 5 test fold
RF_accuracies = cross_val_score(estimator= RFclassifier, X = x_train, y = y_train, cv = 5)
print(" Accuracy for Random Forest after applying KFold cross validation: {:.2f} %".format(RF_accuracies.mean()*100)) # multipling it by 100 to get the value in percentage
# print(" Standard Deviation: {:.2f} %".format(accuracies.std()*100)) # multipling it by 100 to get the value in percentage

''' Applying Gaussian NB '''
#training Naive Bayes model on training dataset
from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(x_train , y_train)

#predicting Test set
y_pred = NBclassifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print()

#making a confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
print("confusion matrix - \n",cm)
print()
ac = accuracy_score(y_test , y_pred)
print("Accuracy Score- \n",ac)
print()

""" Applying K fold cross validation in Naive Bayes """
from sklearn.model_selection import cross_val_score
#accuracy obtained on each of the 5 test fold
NB_accuracies = cross_val_score(estimator= NBclassifier, X = x_train, y = y_train, cv = 5)
print(" Accuracy for Naive Bayes after applying KFold cross validation: {:.2f} %".format(NB_accuracies.mean()*100)) # multipling it by 100 to get the value in percentage
# print(" Standard Deviation: {:.2f} %".format(accuracies.std()*100)) # multipling it by 100 to get the value in percentage

























