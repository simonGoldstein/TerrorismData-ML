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
