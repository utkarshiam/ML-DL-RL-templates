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
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review= review.lower()
review=review.split()
ps= PorterStemmer()
review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review= ' '.join(review)
