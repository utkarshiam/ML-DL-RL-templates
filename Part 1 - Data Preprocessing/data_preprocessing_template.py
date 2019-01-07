# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset= pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y= dataset.iloc[:, 3:4].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer= imputer.fit(X[:,1:3])
X[:, 1:3]= imputer.transform(X[:, 1:3])

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X=LabelEncoder()
X[:, 0]=LabelEncoder_X.fit_transform(X[:, 0])
onehotencoder= OneHotEncoder(categorical_features=[0])
X= onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Test and Training splitting
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=100)

#Let's scale this shite
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

