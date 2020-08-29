# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:23:24 2019

@author: User
"""
from time import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

start = time()
wave = pd.read_csv("waveform.data")

X = wave.iloc[:,:-1].values
y = wave.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

def predict(X_train,y_train,X_test,k):
    distances = []
    targets = []
    for i in range(len(X_train)):
        distances.append([np.sqrt(np.sum(np.square(X_test-X_train[i,:]))),i])
        distances = sorted(distances)
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])
    return Counter(targets).most_common(1)[0][0]

def knn(X_train,y_train,X_test,k):
    predictions = []
    for i in range(len(X_test)):
        predictions.append(predict(X_train,y_train,X_test[i,:],k))
    return np.asarray(predictions)


# making our predictions
predictions = knn(X_train, y_train, X_test, 93)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print("The accuracy of our classifier is {}".format(100*accuracy))
end = time()
print(end-start)