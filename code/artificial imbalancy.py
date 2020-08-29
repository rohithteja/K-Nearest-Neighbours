# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:48:02 2019

@author: User
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.datasets import make_imbalance
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

from collections import Counter

wave = pd.read_csv("waveform.data")

X = wave.iloc[:,:-1].values
y = wave.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

count = Counter(y_train)
print(count)

X_train_imb, y_train_imb = make_imbalance(X_train,y_train,sampling_strategy = {0:1000,1:150,2:2})

accuracy = []
f1 = []
neighbors = list(range(1,51))
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy1 = accuracy_score(y_test,y_pred)
    accuracy.append(accuracy1)
    f12 = f1_score(y_test,y_pred,average='weighted', labels=np.unique(y_pred))
    f1.append(f12)
    
accuracy_imb = []
f1_imb = []
neighbors = list(range(1,51))
for k in neighbors:
    knn_imb = KNeighborsClassifier(n_neighbors=k)
    knn_imb.fit(X_train_imb,y_train_imb)
    y_pred_imb = knn_imb.predict(X_test)
    accuracy1_imb = accuracy_score(y_test,y_pred_imb)
    accuracy_imb.append(accuracy1_imb)
    f12_imb = f1_score(y_test,y_pred_imb,average='weighted', labels=np.unique(y_pred_imb))
    f1_imb.append(f12_imb)

  
plt.plot(neighbors,[x*100 for x in accuracy],linestyle='dashed',marker='o',markerfacecolor='red',label = 'Full data')
plt.plot(neighbors,[x*100 for x in accuracy_imb],linestyle='dashed',marker='o',markerfacecolor='green', label = 'Imbalanced data')
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("imb1.jpg",dpi=500)
plt.show()
    
plt.plot(neighbors, [x*100 for x in f1],linestyle='dashed',marker='o',markerfacecolor='red',label = 'Full data')
plt.plot(neighbors, [x*100 for x in f1_imb],linestyle='dashed',marker='o',markerfacecolor='green',label = 'Imbalanced data')
plt.xlabel("Number of Neighbors K")
plt.ylabel("F1 measure")
plt.legend()
plt.savefig("imb2.jpg",dpi=500)
plt.show()

plt.plot(neighbors,[x*100 for x in f1_imb],linestyle='dashed',marker='o',markerfacecolor='red',label = 'F1 imbalanced')
plt.plot(neighbors, [x*100 for x in accuracy_imb],linestyle='dashed',marker='o',markerfacecolor='green',label = 'Accuracy imbalanced')
plt.xlabel("Number of Neighbors K")
plt.ylabel("F1 measure and Accuracy")
plt.legend()
plt.savefig("imb3.jpg",dpi=500)
plt.show()