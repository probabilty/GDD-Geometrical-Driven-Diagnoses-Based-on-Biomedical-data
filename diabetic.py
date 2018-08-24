
# coding: utf-8

# In[15]:


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


# In[26]:


from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(X_train_sample)
pca.transform(X_train_sample)
pca.transform(X_train_control)
pca.transform(X_test_sample)
pca.transform(X_test_control)


# In[27]:


import sklearn.cluster as cs
def kMeans(X, k):
    cls= cs.k_means(X,n_clusters = k)
    return cls[0]


# In[ ]:


from sklearn.metrics import accuracy_score
from scipy.spatial import distance
old_score = 0.0
score = 0.0
m_opt = 2
n_opt = 2
test_set = np.concatenate((X_test_sample, X_test_control), axis=0)
test_result = np.concatenate((y_test_sample, y_test_control), axis=0)


# In[ ]:

count = 0
for n in range(1, 21):
    centroids_c= kMeans(X_train_control, k = n)
    for m in range(1, 21):
        predicted = []
        distances = []
        centroids_s= kMeans(X_train_sample, k = m)
        centroids = np.concatenate((centroids_s,centroids_c), axis=0)
        i = 0
        for test_case in test_set:
            distances = []
            j = 0
            for c in centroids:
                dst = distance.euclidean(test_set[i],centroids[j])
                distances.extend([dst])
                j = j + 1
            i = i + 1
            if np.argmin(distances) < n:
                predicted.extend([1])
            else:
                predicted.extend([0])
        TP, FP, TN, FN = perf_measure(test_result, predicted)
        #curr_acc = accuracy_score(test_result,predicted)
        if (TP + TN + FP + FN):
            accuracy = (TP + TN)/(TP + TN + FP + FN)
        else:
            accuracy = 0
        if TP + FN:
            sensitivity = TP /(TP + FN)
        else:
            sensitivity = 0
        if TN + FP:
            specificity = TN /(TN + FP)
        else:
            specificity = 0
        if TP + FP:
            positivePredictiveAccuracy = TP /(TP + FP)
        else:
            positivePredictiveAccuracy = 0
        if TN + FN:
            negativePredictiveAccuracy = TN /(TN + FN)
        else:
            negativePredictiveAccuracy = 0
        score = sensitivity + specificity + positivePredictiveAccuracy + negativePredictiveAccuracy
        count = count + 1
        met = {'m':m,'n':n,'score':score,'accuracy' : accuracy, 'Sensitivity' : sensitivity,'Specificity' : specificity,'Positive_Predictive_Accuracy ' : positivePredictiveAccuracy,'Negative_Predictive_Accuracy:' :negativePredictiveAccuracy}
        data = {'postive_centroid':centroids_s,'negative_centroids':centroids_c}
        output = open('data'+str(count)+'.pkl', 'wb')
        pickle.dump(met, output)
        output.close()
        output = open('meta'+str(count)+'.pkl', 'wb')
        pickle.dump(met, output)
        output.close()

        # print (m)
        # print (n)
        # print('score:' + str(score))
        # print('accuracy:' + str(accuracy))
        # print('Sensitivity:' + str(sensitivity))
        # print('Specificity: ' + str(specificity))
        # print('Positive Predictive Accuracy ' + str(positivePredictiveAccuracy))
        # print('Negative Predictive Accuracy:' + str(negativePredictiveAccuracy))
        # print ('---------------------------------------')
        # if score > old_score:
        #     m_opt = m
        #     n_opt = n
        #     old_score = score
        #     output = open('data.pkl', 'ab')
        #     pickle.dump(centroids, output)
        #     output.close()
        #     met = {'m':m,'n':n,'score':score,'accuracy' : accuracy, 'Sensitivity' : sensitivity,'Specificity' : specificity,'Positive_Predictive_Accuracy ' : positivePredictiveAccuracy,'Negative_Predictive_Accuracy:' :negativePredictiveAccuracy}
        #     output = open('meta.pkl', 'ab')
        #     pickle.dump(met, output)
        #     output.close()
        #     print m
        #     print n
        #     print('score:' + str(score))
        #     print('accuracy:' + str(accuracy))
        #     print('Sensitivity:' + str(sensitivity))
        #     print('Specificity: ' + str(specificity))
        #     print('Positive Predictive Accuracy ' + str(positivePredictiveAccuracy))
        #     print('Negative Predictive Accuracy:' + str(negativePredictiveAccuracy))
        #     print ('---------------------------------------')
