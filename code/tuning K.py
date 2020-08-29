# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:03:37 2019

@author: User
"""
import turtle
import pandas as pd
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


wave = pd.read_csv("waveform.data")

X = wave.iloc[:,:-1].values
y = wave.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

cv_scores = []
neighbors = list(range(1,101,2))
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())
  
# changing to misclassification error
mse = [x*100 for x in cv_scores]

# determining best k
optimal_k = neighbors[mse.index(max(mse))]


# plot misclassification error vs k
plt.plot(neighbors, mse,linestyle='dashed',marker='o',markerfacecolor='red')
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.savefig("wave tune.jpg",dpi=500)
plt.show()
    
print("The optimal number of neighbors is {}".format(optimal_k))
knn1 = KNeighborsClassifier(n_neighbors=optimal_k)
knn1.fit(X_train,y_train)
y_pred = knn1.predict(X_test)
print(accuracy_score(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

plt.figure()
sn.set(font_scale=1.4)#for label size
fig = sn.heatmap(confusion_matrix(y_test,y_pred), annot=True,annot_kws={"size": 15}).get_figure()
fig.savefig("heat.jpg",dpi=500)