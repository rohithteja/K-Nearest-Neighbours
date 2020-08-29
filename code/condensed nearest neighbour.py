# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:31:55 2019

@author: User
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import NeighbourhoodCleaningRule
from time import time

wave = pd.read_csv("waveform.data")

X = wave.iloc[:,:-1].values
y = wave.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

cnn = CondensedNearestNeighbour()
X_train_cnn, y_train_cnn = cnn.fit_sample(X_train, y_train)

renn = NeighbourhoodCleaningRule()
X_train_renn,y_train_renn = renn.fit_sample(X_train_cnn,y_train_cnn)
#X_train_renn,y_train_renn = renn.fit_resample(X_train_cnn,y_train_cnn)

accuracy = []
neighbors = list(range(1,11))
for k in neighbors:
    start=time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy1 = accuracy_score(y_test,y_pred)
    accuracy.append(accuracy1)
    end=time()
print("\nTime: ",end-start)

accuracy_cnn = []
for k in neighbors:
    start_cnn=time()
    knn_cnn = KNeighborsClassifier(n_neighbors=k)
    knn_cnn.fit(X_train_cnn,y_train_cnn)
    y_pred_cnn = knn_cnn.predict(X_test)
    accuracy_cnn1 = accuracy_score(y_test,y_pred_cnn)
    accuracy_cnn.append(accuracy_cnn1)
    end_cnn=time()
print("\nTime CNN: ",end_cnn-start_cnn)

accuracy_renn = []
for k in neighbors:
    start_rnn=time()
    knn_renn = KNeighborsClassifier(n_neighbors=k)
    knn_renn.fit(X_train_renn,y_train_renn)
    y_pred_renn = knn_renn.predict(X_test)
    accuracy_renn1 = accuracy_score(y_test,y_pred_renn)
    accuracy_renn.append(accuracy_renn1)
    end_rnn=time()
print("\nTime RNN: ",end_rnn-start_rnn)

# changing to misclassification error
error = [x*100 for x in accuracy]
error_renn = [x*100 for x in accuracy_renn]
error_cnn = [x*100 for x in accuracy_cnn]

# determining best k
optimal_k = neighbors[error.index(min(error))]
optimal_k_renn = neighbors[error_renn.index(min(error_renn))]
optimal_k_cnn = neighbors[error_cnn.index(min(error_cnn))]

# plot misclassification error vs k
plt.plot(neighbors, error,linestyle='dashed',marker='o',color='b',label='Original Dataset')
plt.plot(neighbors, error_renn,linestyle='dashed',marker='o',color='r',label='Using RNN')
plt.plot(neighbors, error_cnn,linestyle='dashed',marker='o',color='g',label='Using CNN')
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("cnn.jpg",dpi=500)
plt.show()
    
#elapsed_time = timeit.timeit(test, number=100)/100
#elapsed_time_cnn = timeit.timeit(test_cnn, number=100)/100
#elapsed_time_renn = timeit.timeit(test_renn, number=100)/100
#print(elapsed_time,elapsed_time_cnn,elapsed_time_renn)