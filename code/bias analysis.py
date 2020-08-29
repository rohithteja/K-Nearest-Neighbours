# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:42:24 2019

@author: User
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

wave = pd.read_csv("waveform.data")

X = wave.iloc[:,:-1].values
y = wave.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.979, random_state = 69, stratify = y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

neighbors = list(range(90,101))
accuracy_k = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_k.append(accuracy)

acc = [x*100 for x in accuracy_k]
plt.plot(neighbors,acc,linestyle='dashed',marker='o',markerfacecolor='red')
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.axhline(86,c='g')

plt.savefig("bias.jpg",dpi=500)
plt.show()    
