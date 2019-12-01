import final_project.py

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
