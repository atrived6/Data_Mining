# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tsfresh.feature_extraction import feature_calculators
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
from random import seed
from random import randrange
import pywt
import sys
import scipy
import joblib


# %%
# Taking the file path from user
testData = pd.read_csv('test.csv', header=None)
testData = testData.dropna()
n_rows = testData.shape[0]
n_col = testData.shape[1]

# Feature Extractions:

# Feature: DWT Transform of data
fft_coef = []
index_ar = []
for i in range(n_rows):
  (d, c) = pywt.dwt(testData.iloc[i,:], 'db2')
  (d, c) = pywt.dwt(d, 'db2')
  fft_coef.append(c)
  index_ar.append(i)

# Feature: Kurtosis

testKurtosis = []
for i in range(n_rows):
  row = testData.iloc[i,:]
  kurtVal = feature_calculators.kurtosis(row)
  testKurtosis.append(kurtVal)

# Feature: LAOGE (Large Amplitude of plasma Glucose Excursions)

testLAOGE = np.zeros(n_rows)
for i in range(n_rows):
  testLAOGE[i] = (np.max(testData.iloc[i,:]) - np.min(testData.iloc[i,:]))

# Feature: COGA (Continuous Overall net Glycemic Action)

testCOGA = []
for i in range(n_rows):
  dayCOGA = []
  for j in range(7, 30):
    if not pd.isnull(testData.iloc[i][j]) and not pd.isnull(testData.iloc[i][j-7]):
      dayCOGA.append(testData.iloc[i][j] - testData.iloc[i][j-7])
  if dayCOGA == []:
    testCOGA.append(0)
  else:
    testCOGA.append(np.std(dayCOGA))


# Feature: Entropy

testEntropy = np.zeros(n_rows)
for i in range(n_rows):
  testEntropy[i] = (scipy.stats.entropy(testData.iloc[i,:]))

# Combining and Normalizing feature matrix

feature_matrix = ['']*n_rows
for i in range(n_rows):
  feature_matrix[i] = np.concatenate((fft_coef[i], np.asarray([testKurtosis[i], testLAOGE[i], testCOGA[i], testEntropy[i]])))
feature_matrix = np.asarray(feature_matrix)

normalized_matrix = preprocessing.normalize(feature_matrix, axis=0, norm='max')

# Applying PCA for top 5 features

pca1 = PCA().fit(normalized_matrix.data)
pca = PCA(n_components=5)#Top 5 Features selected
X_test = pca.fit_transform(normalized_matrix)
np.random.shuffle(X_test)
X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test))

#Loading classifier from the pickle model
clf = joblib.load('Aradhana_SVM.joblib.pkl')
y_pred = clf.predict(X_test)

# Predicting Accuracies from the loaded classier for given testData

for i in range(len(y_pred)):
  meal = 'yes' if y_pred[i] == 1 else 'no'


# %%
np.savetxt("Result.csv", y_pred, delimiter=",")


