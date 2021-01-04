#%%
import csv
import numpy as np
import pandas as pd
from tsfresh.feature_extraction import feature_calculators
from sklearn import preprocessing
from sklearn.decomposition import PCA
from random import seed
from random import randrange
import pywt
import scipy
import math
import sklearn

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# %%
def get_meal_nomealdata(cgm_df, insulin_df):
  meal_intake_df = insulin_df[insulin_df['BWZ Carb Input (grams)'].notnull()]
  meal_intake_data = pd.DataFrame()
  for i in range(len(meal_intake_df.index)-1):
    if(pd.Timedelta(meal_intake_df.iloc[i+1,meal_intake_df.columns.get_loc("DateTime")]-meal_intake_df.iloc[i,meal_intake_df.columns.get_loc("DateTime")]).seconds/3600 > 2):
      meal_intake_data = pd.concat([meal_intake_data,meal_intake_df.iloc[[i]]])

  meal_intake_data = pd.concat([meal_intake_data,meal_intake_df.iloc[[len(meal_intake_df.index)-1]]])
  
  
  meal_data = []
  carb_data=[]
  for i in range(len(meal_intake_data.index)-1):
    time = meal_intake_data.iloc[i,meal_intake_data.columns.get_loc("DateTime")]
    start = time
    end = time + pd.Timedelta(hours=2)
    data = cgm_df[(cgm_df['DateTime']>=start) & (cgm_df['DateTime']<=end)] ['Sensor Glucose (mg/dL)']
    data = data.dropna()
    if len(data)==24:
      meal_data.append(data)
      carb_data.append(meal_intake_data.iloc[i,meal_intake_data.columns.get_loc("BWZ Carb Input (grams)")])
    
    

  return meal_data,carb_data


  
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
patient_1_data,carbs = get_meal_nomealdata(reversed_cgmData,reversed_insulinData)
#%%  
meal_data_frame=[]

for x in patient_1_data:
    meal_data_frame.append(x.to_list())
meal_data_len = len(patient_1_data)
meal_data_frame = pd.DataFrame(meal_data_frame)

# %%
fft_coef = []
index_ar = []
for i in range(len(meal_data_frame)):
  (d, c) = pywt.dwt(meal_data_frame.iloc[i,:], 'db2')
  (d, c) = pywt.dwt(d, 'db2')
  fft_coef.append(c)
  index_ar.append(i)

# %%
# Feature: Kurtosis

MealKurtosis = []
for i in range(meal_data_len):
  row = meal_data_frame.iloc[i,:]
  kurtVal = feature_calculators.kurtosis(row)
  MealKurtosis.append(kurtVal)
# %%
Meal_Amp = np.zeros(meal_data_len)
for i in range(meal_data_len):
  Meal_Amp[i] = (np.max(meal_data_frame.iloc[i,:]) - np.min(meal_data_frame.iloc[i,:]))
# %%
Meal_diff = []
for i in range(meal_data_len):
  dayCOGA = []
  for j in range(7, 24):
    if not pd.isnull(meal_data_frame.iloc[i][j]) and not pd.isnull(meal_data_frame.iloc[i][j-7]):
      dayCOGA.append(meal_data_frame.iloc[i][j] - meal_data_frame.iloc[i][j-7])
  if dayCOGA == []:
    Meal_diff.append(0)
  else:
    Meal_diff.append(np.std(dayCOGA))
# %%
# Entropy calculation
Meal_Entropy = np.zeros(meal_data_len)
for i in range(meal_data_len):
  Meal_Entropy[i] = (scipy.stats.entropy(meal_data_frame.iloc[i,:]))
# %%
feature_matrix = ['']*meal_data_len
for i in range(meal_data_len):
  feature_matrix[i] = np.concatenate((fft_coef[i], np.asarray([MealKurtosis[i], Meal_Amp[i], Meal_diff[i], Meal_Entropy[i]])))
feature_matrix = np.asarray(feature_matrix)
normalized_matrix = preprocessing.normalize(feature_matrix, axis=0, norm='max')
# %%
pca_varience = PCA().fit(normalized_matrix.data)
pca = PCA(n_components=5)
data = pca.fit_transform(normalized_matrix)
# %%
carbs_labels = []
for i in range(len(data)):
    if carbs[i] > 0 and carbs[i] <= 20:
        carbs_labels.append(0)
    elif carbs[i] > 20 and carbs[i] <= 40:
        carbs_labels.append(1)
    elif carbs[i] > 40 and carbs[i] <= 60:
        carbs_labels.append(2)
    elif carbs[i] > 60 and carbs[i] <= 80:
        carbs_labels.append(3)
    elif carbs[i] > 80 and carbs[i] <= 100:
        carbs_labels.append(4)
    else:
        carbs_labels.append(5)
# %%
max1 = max2 = 0
# stored_tr_X, s
for i in range(10):
  km = KMeans(n_clusters=6, init="k-means++", tol=0.0001).fit(normalized_matrix)
  y_label_km = km.labels_
  accuracy_km = sklearn.metrics.accuracy_score(carbs_labels, y_label_km)
  if accuracy_km > max1:
    max1 = accuracy_km
    ari1 = sklearn.metrics.adjusted_rand_score(carbs_labels, y_label_km)
    SSE_score_km = mean_squared_error(carbs_labels, y_label_km)

# %%
conv = metrics.cluster.contingency_matrix(carbs_labels, y_label_km)

# %%
total = len(carbs_labels)
entropy_km= 0
for x in range(len(conv)):
    local_total = sum(conv[x])
    local_entropy = 0
    for i in range(len(conv[x])):
        local_entropy = local_entropy - ((conv[x][i]/local_total)*math.log((conv[x][i]/local_total), 2))
    entropy_km = entropy_km + (local_entropy*(local_total/total))

# %%
purity_km =  np.sum(np.amax(conv, axis=0)) / np.sum(conv)
# %%
print("SSE: {}, Entropy: {}, Purity: {}".format(SSE_score_km, entropy_km, purity_km))
# %%
for i in range(10):
    cluster = AgglomerativeClustering(n_clusters = 6, affinity='euclidean', linkage='ward')
    cluster.fit(normalized_matrix)
    y_label_db = cluster.labels_
    h_clusters_df = pd.DataFrame(y_label_db)
    data = StandardScaler().fit_transform(h_clusters_df)
    db = DBSCAN(eps = 0.05, min_samples = 2).fit(data)
    y_label_db = db.labels_
    n_clusters_ = len(set(y_label_db)) - (1 if -1 in y_label_db else 0)
    accuracy_db = sklearn.metrics.accuracy_score(carbs_labels, y_label_db)
    cluster_score = sklearn.metrics.adjusted_rand_score(carbs_labels, y_label_db)
    if accuracy_db > max2:
        max2 = accuracy_db
        SSE_score_db = mean_squared_error(carbs_labels, y_label_db)
# %%
conv = metrics.cluster.contingency_matrix(carbs_labels, y_label_db)
total = len(carbs_labels)
entropy_db= 0
for x in range(len(conv)):
    local_total = sum(conv[x])
    local_entropy = 0
    for i in range(len(conv[x])):
        local_entropy = local_entropy - ((conv[x][i]/local_total)*math.log((conv[x][i]/local_total), 2))
    entropy_db = entropy_db + (local_entropy*(local_total/total))
purity_db =  np.sum(np.amax(conv, axis=0)) / np.sum(conv)
print("SSE: {}, Entropy: {}, Purity: {}".format(SSE_score_db, entropy_db, purity_db))

# %%
np_array = np.array([SSE_score_km, SSE_score_db, entropy_km, entropy_db, purity_km, purity_db])
np_array.tofile('Results.csv', sep = ',')
# %%
