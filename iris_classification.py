# -*- coding: utf-8 -*-
"""
Created on Sun July 27 10:52:08 2018

@author: bhuvan
"""

#importing libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#importing the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('iris.csv', names = names)

#peeking into the data
#shape
print(dataset.shape)
#head
print(dataset.head(20))

#getting the statistical summary of the data
print(dataset.describe())

#class distribution
print(dataset.groupby('class').size())

#visualzing the data
#univariate plots
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

#mutivariate plots to observe relation betweend features
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

#preparing the training and test datasets
#Split-out validation dataset
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values
validation_size = 0.20
seed = 7
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

#Test options and evaluation metric
seed = 7
scoring = 'accuracy'

#Using multiple algorithms to compare the results and find the best model.
#Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	model_output = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(model_output)
    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#since KNN is the best model for our data
#use the KNN model on the validation or test dataset
#make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
