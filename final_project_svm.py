import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import os
currentPath = os.path.dirname(os.path.realpath(__file__))

plt.style.use("ggplot")
dataset = pd.read_csv(os.path.join(currentPath, 'data/globalterrorismdb_0718dist.csv') ,encoding='ISO-8859-1' )

print(dataset.head())
print(dataset.describe())
#Create X and y
y = dataset[['gname']] #, 'gsubname', 'gname2', 'gsubname2','gname3', 'gsubname3', 'crit1', 'crit2', 'crit3']].copy()
X = dataset.copy()

# Mannually Drop bad columns
# Y columns
X.drop(columns = ['gname', 'gsubname', 'gname2', 'gsubname2','gname3', 'gsubname3', 'crit1', 'crit2', 'crit3'], inplace=True)
# Citations
X.drop(columns = ['scite1', 'scite2', 'scite3', 'dbsource'], inplace=True)
# anything_txt
X.drop(columns = ['country_txt', 'region_txt', 'alternative_txt', 'attacktype1_txt', 'attacktype2_txt', 'attacktype3_txt', 'targtype1_txt', 'hostkidoutcome_txt', 'targsubtype1_txt', 'natlty1_txt', 'natlty2_txt', 'natlty3_txt', 'targtype2_txt', 'targsubtype2_txt', 'targtype3_txt', 'targsubtype3_txt', 'propextent_txt', 'claimmode_txt', 'claimmode2_txt', 'claimmode3_txt', 'weaptype1_txt', 'weapsubtype1_txt', 'weaptype2_txt', 'weapsubtype2_txt', 'weaptype3_txt', 'weapsubtype3_txt', 'weaptype4_txt', 'weapsubtype4_txt'], inplace=True)
# The kitchen sink
X.drop(columns = ['ransomnote', 'eventid', 'resolution', 'approxdate', 'location', 'summary', 'city', 'provstate', 'corp1', 'target1', 'corp2', 'target2', 'corp3', 'target3', 'motive', 'weapdetail', 'doubtterr', 'related', 'propcomment', 'divert', 'kidhijcountry', 'addnotes'], inplace=True)

#Find columns with NaN's and those without
#print(dataset.isnull().sum())\
fullNames = []
emptyNames = []
for (colName, colData) in X.iteritems():
  #print(colName)
  #print(colData.isnull().sum())
  if colData.isnull().sum() == 0:
    fullNames.append(colName)
  else:
    emptyNames.append(colName)



X.drop(emptyNames, inplace = True, axis = 1)

X.describe()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
X_model, X_valid, y_model, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.2)

number = 200
X_train = X_train[:][1:number]
y_train = y_train[:][1:number]

print("Training")

"""
hyperparams = {
    "kernel": ["linear"],
    "degree": [1, 2, 3],
    "C": [1e-2, 1e-1, 1, 1e2],
    "random_state": [0]
}

svc = sk.svm.SVC()
clf = GridSearchCV( svc, hyperparams, "accuracy" )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Linear:")
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred, average='micro'))
print("model params: ", clf.best_estimator_)
"""
svc = sk.svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=0,
    shrinking=True, tol=0.001, verbose=False)
svc.fit(X_train, y_train)

import helpers.modelSaving as save
save.saveSk(svc, f'trainedModels/svm-{number}')

y_pred = svc.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred, average='weighted'))

clf = sk.svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=0,
    shrinking=True, tol=0.001, verbose=False)
clf.fit(X_model, y_model)
validationPredictions = clf.predict(X_valid)
print(confusion_matrix(y_valid, validationPredictions))
print(classification_report(y_valid, validationPredictions))

"""
hyperparams = {
    "kernel": ["rbf"],
    "degree": [1, 2, 3],
    "C": [1e-2, 1e-1, 1, 1e2],
    "random_state": [0]
}

svc = svm.SVC(kernel='rbf', gamma=0.7)
clf = GridSearchCV( svc, hyperparams, "accuracy" )
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)
print("RBF:")
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred, average='micro'))

hyperparams = {
    "kernel": ["poly"],
    "degree": [1, 2, 3],
    "C": [1e-2, 1e-1, 1, 1e2],
    "random_state": [0],
    "gamma": ["scale"]
}

svc = svm.SVC(kernel='poly', degree=3)
clf = GridSearchCV( svc, hyperparams, "accuracy" )
clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)
print("Poly:")
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
### No precision metrics were accepted without error, only on this one weird
#print("Precision: ", metrics.precision_score(y_test, y_pred), average='weighted')
"""
