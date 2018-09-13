#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 00:42:58 2018

@author: dohoonkim
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc

#data_features = ["Alpha2","Beta0","Beta1","Beta2","Beta3","Delta0","Delta1","Delta2","Delta3","Delta5","Low0","Low1","Low2","Low3","Low5"]
data_features = ["Beta0","Beta1","Beta2","Beta3","Delta0","Delta1","Delta2","Delta3","Delta5","Low0","Low1","Low2","Low3","Low5"]
#data_features = ["FractionAvg", "Efficiency", "FractionHigh"]

Insomnia_PATH = "/Users/dohoonkim/Desktop/Research/insomnia/insomnia2000noalpha.csv"
Normal_PATH = "/Users/dohoonkim/Desktop/Research/insomnia/normal2000noalpha.csv"
#raw_PATH= "/Users/dohoonkim/Desktop/Research/insomnia/data1.csv"
insomnia_data = pd.read_csv(Insomnia_PATH)
normal_data = pd.read_csv(Normal_PATH)
whole_data = pd.concat([insomnia_data,normal_data])


#raw_data = pd.read_csv(raw_PATH)
#scaler = MinMaxScaler(feature_range=(0, 1))
#normalized_scaled_data = StandardScaler().fit_transform(data[data_features])
#datacor = data[data_features]
#cor = datacor.corr()
#sns.heatmap(cor, square = True)
#shuffled = whole_data.reset_index()
#shuffled.to_csv("/Users/dohoonkim/Desktop/Research/insomnia/shuffled1.csv",index=False)
Data_to_use = pd.read_csv("/Users/dohoonkim/Desktop/Research/insomnia/shuffled1.csv")
train_x, test_x, train_y, test_y = train_test_split(Data_to_use[data_features],Data_to_use["Insomnia"], train_size=0.7)

#PCA version:
#pca = PCA().fit(train_x)
pca = PCA(n_components=4)
#data_4d = pca.fit_transform(Data_to_use[data_features])
pca = pca.fit(train_x)
train_4d = pca.transform(train_x)
test_4d =  pca.transform(test_x)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');
#y_score=[]
#for i in range(0,10):
#    clf = AdaBoostClassifier(n_estimators=2)
#    fit = clf.fit(train_4d,train_y)
#    if i == 0:
#        y_score = fit.decision_function(test_4d)
#
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#test_y_array= test_y.as_matrix(columns=None)
#
#fpr, tpr, threshold = roc_curve(test_y_array, y_score, pos_label=1)
#roc_auc = auc(fpr, tpr)
#plt.figure()
#lw = 2
#plt.plot(fpr, tpr, color='green',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC curve for Adaboost')
#plt.legend(loc="lower right")
#plt.show()
#we get approximately 100percent and diminishing return of explained variance at number of components to be 4




#clf = AdaBoostClassifier(n_estimators=2)

#fit = clf.fit(train_4d,train_y)
#scores = cross_val_score(clf, data_4d, Data_to_use["Insomnia"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#first implement with logistic regression
#lr = linear_model.LogisticRegression(fit_intercept=True)
#lr_fit = lr.fit(train_4d,train_y)
##
#print('Binary Logistic regression Train Accuracy :: {}'.format(metrics.accuracy_score(train_y, lr_fit.predict(train_4d))))
#print('Binary Logistic regression Test Accuracy :: {}'.format(metrics.accuracy_score(test_y, lr_fit.predict(test_4d))))
#scores = cross_val_score(lr, data_4d, Data_to_use["Insomnia"], cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#

#scaled_data = scaler.fit_transform(data[data_features])
#normalized_raw_data = StandardScaler().fit_transform(raw_data[data_features])

#2-means clustering to visualize how data are centered
#kmeans = KMeans(n_clusters=2, random_state= 40, n_init=10).fit(scaled_data)
#centroids = kmeans.cluster_centers_

