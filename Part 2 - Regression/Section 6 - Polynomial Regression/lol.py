#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 03:11:21 2018

@author: mudita
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree= 6)
X_poly= poly_reg.fit_transform(X)

lin_reg1= LinearRegression()
lin_reg1=lin_reg1.fit(X_poly,y) 


#plot Linear regression
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color='blue')
plt.show()
#plot polynomial regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg1.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()
lin_reg1.predict(poly_reg.fit_transform(6.5))