# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)


# Cleaning the texts
import re
import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
lol=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review= review.lower()
    review=review.split()
    ps= PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    lol.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(lol).toarray()
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#ANY ClASSIFIER YOU WANT TO APPLY!!!


#ACCURACY
from sklearn.metrics import accuracy_score

predictions = classifier.predict(X_test)

print(accuracy_score(y_test, predictions))