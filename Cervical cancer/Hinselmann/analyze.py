import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.cluster as cs
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle
from sklearn import (cross_validation, feature_selection, pipeline,
                     preprocessing, linear_model, grid_search)
import pandas as pd
dataset = pd.read_csv('Data.csv')
dataset.head()
target = 'Hinselmann'
var = dataset.var().sort_values()
dataset = dataset.fillna(method='pad')



sample = dataset[dataset[target] == 1]
control = dataset[dataset[target] == 0]
X_sample = sample.iloc[:, sample.columns != target].values
y_sample = sample.iloc[:, sample.columns ==target].values
X_control = control.iloc[:, control.columns != target].values
y_control = control.iloc[:, control.columns ==target].values
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_sample, y_sample, test_size = 0.25, random_state = 0)
X_train_control, X_test_control, y_train_control, y_test_control = train_test_split(X_control, y_control, test_size = 0.25, random_state = 0)

sc = StandardScaler()
sc.fit(np.vstack((X_train_sample,X_train_control)))
X_test_sample = sc.transform(X_test_sample)
X_test_control = sc.transform(X_test_control)
X_test_sample = sc.transform(X_train_sample)
X_train_control = sc.transform(X_train_control)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(np.vstack((X_train_control,X_train_sample)))
X_train_sample = pca.transform(X_train_sample)
X_train_control = pca.transform(X_train_control)
X_test_sample = pca.transform(X_test_sample)
X_test_control = pca.transform(X_test_control)


X_train_control = np.transpose(X_train_control)
X_train_sample = np.transpose(X_train_sample)
import matplotlib.pyplot as plt
plt.scatter(X_train_sample[0], X_train_sample[1], c='blue', alpha=0.5)
plt.scatter(X_train_control[0], X_train_control[1], c='red', alpha=0.5)
plt.show()