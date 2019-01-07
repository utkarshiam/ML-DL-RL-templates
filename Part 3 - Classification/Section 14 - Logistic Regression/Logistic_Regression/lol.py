#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 23:12:19 2018

@author: mudita
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasetdat
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


#split dataset 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=0)

#Scaling Data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

 #Fit the set into logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
 
y_pred=classifier.predict(X_test)

 
 #Construct a confusion matrix
from  sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
 