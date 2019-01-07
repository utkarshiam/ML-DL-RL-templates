#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 01:32:36 2018

@author: mudita
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
X= dataset.iloc[:,:4]
Y= dataset.iloc[:, 4:5]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder= LabelEncoder()
 
