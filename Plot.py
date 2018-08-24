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

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)


# In[16]:


dataset = pd.read_csv('diabetic_dataa.csv')



# In[17]:


dataset = pd.get_dummies(dataset)


# In[18]:


sample = dataset[dataset['diabetesMed'] == 1]
control = dataset[dataset['diabetesMed'] == 0]



# In[19]:


X_sample = sample.iloc[:, sample.columns != 'diabetesMed'].values
y_sample = sample.iloc[:, sample.columns =='diabetesMed'].values
X_control = control.iloc[:, control.columns != 'diabetesMed'].values
y_control = control.iloc[:, control.columns =='diabetesMed'].values



# In[21]:


X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_sample, y_sample, test_size = 0.25, random_state = 0)
X_train_control, X_test_control, y_train_control, y_test_control = train_test_split(X_control, y_control, test_size = 0.25, random_state = 0)


# In[22]:


for i in range(2):
    X_train_sample = np.concatenate((X_train_sample,X_train_sample), axis = 0)
    X_test_sample = np.concatenate((X_test_sample,X_test_sample), axis = 0)
    y_train_sample = np.concatenate((y_train_sample,y_train_sample), axis = 0)
    y_test_sample = np.concatenate((y_test_sample,y_test_sample), axis = 0)


# In[23]:


sc = StandardScaler()
X_train_sample = sc.fit_transform(X_train_sample)
X_train_control = sc.fit_transform(X_train_control)
X_test_sample = sc.fit_transform(X_test_sample)
X_test_control = sc.fit_transform(X_test_control)


# In[24]:


X = dataset.iloc[:, sample.columns != 'diabetesMed'].values
y = dataset.iloc[:, sample.columns == 'diabetesMed'].values


# In[25]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier()
clf = clf.fit(X,y)
model = SelectFromModel(clf, prefit=True)
X_train_sample = model.transform(X_train_sample)
X_train_control = model.transform(X_train_control)
X_test_sample = model.transform(X_test_sample)

X_test_control = model.transform(X_test_control)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
X =[]
Y = []
for f in range(10):
    Y.append(importances[indices[f]])
    X.append(dataset.columns.get_values()[indices[f]])
import matplotlib.pyplot as plt
plt.figure()
plt.title("Feature importances")
plt.bar(X, Y,color="r", align="center")
#plt.xticks(X, Y)
plt.xticks(rotation=90)
plt.savefig('filename.png',dpi = 1000)
plt.show()
# Plot the feature importances of the forest
plt.show()