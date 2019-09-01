#Machine Learning Assignment 1
#uses Iris database from sklearn and SGD classifier

#Reference - G&M_SGD_Classifier.py, provided by Prof. Murphy
#Reference - Google Search

import sklearn
print( 'The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn import datasets
iris           = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

print( "Dimension of entire dataset=", X_iris.shape, y_iris.shape, '\n')

print( "Examples", '\n', X_iris[0:5], y_iris[0:5], '\n')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Dataset with only the first two attributes
X, y    = X_iris[:, :2], y_iris

# Splitting the dataset randomly into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)
print( "training set dimension =", X_train.shape, y_train.shape, '\n')

#Standardising the features
scaler  = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

#Plotting sepal length and sepal width for all three types
import matplotlib.pyplot as plt
colors  = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs  = X_train[:, 0][y_train == i]
    ys  = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

#Importing Classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()

#Fitting the model to training set
clf.fit(X_train, y_train)

print("model parameters =", '\n', clf.coef_,'\n')

print("model intercepts =", '\n', clf.intercept_,'\n')

import numpy as np
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5

Xs = np.arange(x_min, x_max, 0.5)

fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    Ys = (-clf.intercept_[i] - Xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(Xs, Ys)
    
print("sample(4.7,3.1) predict =", clf.predict(scaler.transform([[4.7, 3.1]])),'\n' )

print("decision function =", clf.decision_function(scaler.transform([[4.7, 3.1]])),'\n' )

#predicting for training set and performance measure
from sklearn import metrics
y_train_pred = clf.predict(X_train)
print("Accuracy_score_training =", metrics.accuracy_score(y_train, y_train_pred),'\n' )


y_pred = clf.predict(X_test)
print("Accuracy_score_testing =", metrics.accuracy_score(y_test, y_pred),'\n' )


print("classification report", '\n', metrics.classification_report(y_test, y_pred, target_names=iris.target_names),'\n' )


print("Confusion Matrix", '\n', metrics.confusion_matrix(y_test, y_pred),'\n' )


print( "My name is Rakesh Reddy Mudhireddy" )
print( "My NetID is rmudhi2" )
print( "I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



#######End###########
