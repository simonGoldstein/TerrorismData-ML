import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression, LassoCV

plt.style.use("ggplot")
dataset = pd.read_csv('used_data/globalterrorismdb_0718dist.csv' ,encoding='ISO-8859-1' )

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

"""#### PCA!!!!"""

#pca_tf = sk.decomposition.PCA(n_components=50)
#pca_tf.fit(X)
#X_pca = pca_tf.transform(X)

"""#### Split data into training, testing, and validataion sets"""

#split data into training, testing, and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
X_model, X_valid, y_model, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.2)

#print(f"All Data:        {len(X)} points")
#print(f"Training data:   {len(X_train)} points")
#print(f"Testing data:    {len(X_test)} points")
#print(f"Modeling data:   {len(X_model)} points")
#print(f"Validation data: {len(X_valid)} points")

"""### Model TIME!!!"""

k = 5

#TODO use OneHotEncoder to fix this error?
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# Step 1 - Initialize model with parameters
knn = KNeighborsClassifier(n_neighbors=k)
# Step 2 - Fit the model data
knn.fit(X_model, y_model)
# Step 3 - Predict the validation data
validationPredictions = knn.predict(X_valid)

"""Generate Model"""

#validation step
print(confusion_matrix(y_valid, validationPredictions))
print(classification_report(y_valid,validationPredictions))
f1_score(y_valid, validationPredictions, average="weighted")

def get_knn_training_scores(ks, model_features, model_labels):
    """Determine the f1-score of k values for kNN on a given data set
    Args:
        ks (int iterable): iterable of all the k values to apply
        model_features (iterable): the features from the model set to train on
        model_labels (iterable): the labels from the model set to train on

    Returns:
        dictionary: key is the k value and value is the weighted f1_score on the training set
    """
    dict = {}

    for k in ks:
      # Step 1 - Initialize model with parameters
      knn = KNeighborsClassifier(n_neighbors=k)
      # Step 2 - Fit the model data
      knn.fit(model_features, model_labels)
      # Step 3 - Predict the validation data
      validationPredictions = knn.predict(model_features)

      dict[k] = f1_score(model_labels, validationPredictions, average="weighted")

    return dict

def get_knn_validation_scores(ks, model_features, model_labels, validation_features, validation_labels):
    """Train a model on a dataset then return the F-1 score on another set
    Args:
        ks (int iterable): iterable of all the k values to apply
        model_features (iterable): the features from the model set to train on
        model_labels (iterable): the labels from the model set to train on
        validation_features (iterable): the features from the validation set to test on
        validation_labels (iterable): the labels from the validation set to test on

    Returns:
        dictionary: key is the k value and value is the weighted f1_score on the validation set
    """
    dict = {}

    for k in ks:
      # Step 1 - Initialize model with parameters
      knn = KNeighborsClassifier(n_neighbors=k)
      # Step 2 - Fit the model data
      knn.fit(model_features, model_labels)
      # Step 3 - Predict the validation data
      validationPredictions = knn.predict(model_features)

      dict[k] = f1_score(model_labels, validationPredictions, average="weighted")

    return dict

#find best k
ksToTest = [1,3,5,7,10,20,50,100]
training_scores = get_knn_training_scores(ksToTest, X_model, y_model)
validation_scores = get_knn_validation_scores(ksToTest, X_model, y_model, X_valid, y_valid)

#plot stuff
pd.Series(training_scores, name="Training").plot(kind="line")
pd.Series(validation_scores, name="Validation").plot(kind="line", label="Validation")
plt.legend()
plt.xlabel("k")
plt.ylabel("F1-score")
plt.show()
