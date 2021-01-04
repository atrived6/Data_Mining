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
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


# %%
def get_meal_nomealdata(cgm_df, insulin_df):
  meal_intake_df = insulin_df[insulin_df['BWZ Carb Input (grams)'].notnull()]
  meal_intake_data = pd.DataFrame()
  noMeal_intake_data = pd.DataFrame()
  for i in range(len(meal_intake_df.index)-1):
    if(pd.Timedelta(meal_intake_df.iloc[i+1,meal_intake_df.columns.get_loc("DateTime")]-meal_intake_df.iloc[i,meal_intake_df.columns.get_loc("DateTime")]).seconds/3600 > 2):
      meal_intake_data = pd.concat([meal_intake_data,meal_intake_df.iloc[[i]]])

  meal_intake_data = pd.concat([meal_intake_data,meal_intake_df.iloc[[len(meal_intake_df.index)-1]]])

  noMeal_intake_data = pd.concat([noMeal_intake_data,meal_intake_df.iloc[[0]]])
  for i in range(len(meal_intake_df.index)-1):
    if(pd.Timedelta(meal_intake_df.iloc[i+1,meal_intake_df.columns.get_loc("DateTime")]-meal_intake_df.iloc[i,meal_intake_df.columns.get_loc("DateTime")]).seconds/3600 > 4):
      noMeal_intake_data = pd.concat([noMeal_intake_data,meal_intake_df.iloc[[i+1]]])
  
 
  meal_data = []
  noMeal_data = []
  for i in range(len(meal_intake_data.index)-1):
    time = meal_intake_data.iloc[i,meal_intake_data.columns.get_loc("DateTime")]
    start = time
    end = time + pd.Timedelta(hours=2)
    data = cgm_df[(cgm_df['DateTime']>=start) & (cgm_df['DateTime']<=end)] ['Sensor Glucose (mg/dL)']
    data = data.dropna()
    if len(data)==24:
      meal_data.append(data)

    
  for i in range(len(noMeal_intake_data.index)-1):
    time = noMeal_intake_data.iloc[i,noMeal_intake_data.columns.get_loc("DateTime")]
    start = time+pd.Timedelta(hours=2)
    end = start+pd.Timedelta(hours=2)
    nodata = cgm_df.loc[(cgm_df['DateTime']>=start) & (cgm_df['DateTime']<=end)] ['Sensor Glucose (mg/dL)']
    nodata = nodata.dropna()
    if len(nodata)==24:
      noMeal_data.append(nodata)


  return [meal_data, noMeal_data]


# %%
cgm_data = pd.read_csv("CGMData.csv")
cgm_df = cgm_data[['Date' ,'Time', 'Sensor Glucose (mg/dL)']]
insulin_data = pd.read_csv("InsulinData.csv")
insulin_df = insulin_data[['Date', 'Time', 'BWZ Carb Input (grams)']]
reversed_cgmData = cgm_df.iloc[: :-1]
reversed_insulinData = insulin_df.iloc[: :-1]
reversed_cgmData['DateTime'] = pd.to_datetime(reversed_cgmData['Date']+' '+ reversed_cgmData['Time'])
reversed_insulinData['DateTime'] = pd.to_datetime(reversed_insulinData['Date']+' '+ reversed_insulinData['Time'])
reversed_cgmData['Sensor Glucose (mg/dL)'].fillna(method='ffill', inplace=True)
patient_1_data = get_meal_nomealdata(reversed_cgmData,reversed_insulinData)


# %%
insulin_data1 = pd.read_excel('InsulinAndMealIntake670GPatient3.xlsx')
cgm_data1 = pd.read_excel('CGMData670GPatient3.xlsx')
insulin_data1.to_csv('./InsulinData2.csv')
cgm_data1.to_csv('./CGMData2.csv')
insulin_data1 = pd.read_csv('./InsulinData2.csv')
cgm_data1 = pd.read_csv('./CGMData2.csv')
cgm_data1['DateTime'] = pd.to_datetime(cgm_data1['Date']+' '+ cgm_data1['Time'])
insulin_data1['DateTime'] = pd.to_datetime(insulin_data1['Date']+' '+ insulin_data1['Time'])
patient_2_data = get_meal_nomealdata(cgm_data1, insulin_data1)


# %%
# Combining Data
p1 = patient_1_data[0][0].tolist()
xy = []
for x in patient_1_data[0]:
  xy.append(x.to_list())

for x in patient_2_data[0]:
  xy.append(x.tolist())

meal_data_len = len(xy)

for x in patient_1_data[1]:
  xy.append(x.to_list())

for x in patient_2_data[1]:
  xy.append(x.to_list())

nomeal_data_len = len(xy) - meal_data_len

combined_data = pd.DataFrame(xy)
combined_data_len = len(xy)

# Fourier Transform
fft_coef = []
index_ar = []
for i in range(len(xy)):
  (d, c) = pywt.dwt(combined_data.iloc[i,:], 'db2')
  (d, c) = pywt.dwt(d, 'db2')
  fft_coef.append(c)
  index_ar.append(i)

# Feature: Kurtosis

MealKurtosis = []
for i in range(combined_data_len):
  row = combined_data.iloc[i,:]
  kurtVal = feature_calculators.kurtosis(row)
  MealKurtosis.append(kurtVal)


# %%
# Amplitude diff
Meal_Amp = np.zeros(combined_data_len)
for i in range(meal_data_len):
  Meal_Amp[i] = (np.max(combined_data.iloc[i,:]) - np.min(combined_data.iloc[i,:]))


# %%
# Calculating meal diff values based on meal, nomeal data
Meal_diff = []
for i in range(combined_data_len):
  dayCOGA = []
  for j in range(7, 24):
    if not pd.isnull(combined_data.iloc[i][j]) and not pd.isnull(combined_data.iloc[i][j-7]):
      dayCOGA.append(combined_data.iloc[i][j] - combined_data.iloc[i][j-7])
  if dayCOGA == []:
    Meal_diff.append(0)
  else:
    Meal_diff.append(np.std(dayCOGA))


# %%
# Entropy calculation
Meal_Entropy = np.zeros(combined_data_len)
for i in range(combined_data_len):
  Meal_Entropy[i] = (scipy.stats.entropy(combined_data.iloc[i,:]))


# %%
# Combining and Normalizing feature matrix
feature_matrix = ['']*combined_data_len
for i in range(combined_data_len):
  feature_matrix[i] = np.concatenate((fft_coef[i], np.asarray([MealKurtosis[i], Meal_Amp[i], Meal_diff[i], Meal_Entropy[i]])))
feature_matrix = np.asarray(feature_matrix)
normalized_matrix = preprocessing.normalize(feature_matrix, axis=0, norm='max')


# %%
# PCA
pca_varience = PCA().fit(normalized_matrix.data)
pca = PCA(n_components=5)
data = pca.fit_transform(normalized_matrix)


# %%
# Adding class labels for data
updated_normalized_matrix = np.zeros((combined_data_len, 6))
for i in range(combined_data_len):
  if i < meal_data_len:
    updated_normalized_matrix[i] = np.append(data[i],1)
  else:
    updated_normalized_matrix[i] = np.append(data[i],0)


# %%
# Cross Validation Split for K- Fold validation
def cv_split(dataset, num_folds):
  copy = list(dataset)
  split = list()
  f_s = int(len(dataset)/num_folds)
 
  for i in range(num_folds):
    f = list()
    while len(f) < f_s:
      index = randrange(len(copy))
      f.append(copy.pop(index))
    split.append(f)

  return split


# %%
# Computing confusion matrix for all splits
num_folds = 10
folds = np.asarray(cv_split(updated_normalized_matrix, num_folds))

accuracy = list()
f1 = list()
acc = 0
for i in range(num_folds):
  X_train = []
  y_train = []
  X_test = []
  y_test = []

  for j in range(num_folds):
    if(j!=i):
      for l in range(len(folds[j])):
        X_train.append(folds[j,l,range(0,5)])
        y_train.append(folds[j,l,5])
  for j in range(len(folds[i])):
    X_test.append(folds[i,j,range(0,5)])
    y_test.append(folds[i,j,5])
  X_train,y_train,X_test,y_test = np.asarray(X_train),np.asarray(y_train),np.asarray(X_test),np.asarray(y_test)
  X_train = np.hstack((np.matrix(np.ones(X_train.shape[0])).T, X_train)) 
  X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test)) 
  clf = SVC(gamma='auto') 
  clf.fit(X_train,y_train) 
  y_pred = clf.predict(X_test)
  cur_acc = accuracy_score(y_test, y_pred)
  accuracy.append(cur_acc)
  f1.append(f1_score(y_test, y_pred))
  print("Accuracy Report for fold: "+str(i+1)+"\n", classification_report(y_test, y_pred))
print("Average accuracy: " + str(np.mean(accuracy)))
print("Average F1 score:" + str(np.mean(f1)))


# %%
X_fit=[]
Y_fit=[]
for row in updated_normalized_matrix:
    row = np.insert(row,0,1)
    X_fit.append(row[0:6])
    Y_fit.append(row[6])
clf = SVC(gamma='auto')
clf.fit(X_fit, Y_fit)
filename='Aradhana_SVM.joblib.pkl'
_ = joblib.dump(clf, filename, compress=9)


# %%



