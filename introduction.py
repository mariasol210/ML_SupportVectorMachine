# https://archive.ics.uci.edu/ml/datasets.php
import numpy as np 
import pandas as pd 
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split # Esto porque cross_validation ya no funciona

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1)) # Features
y = np.array(df['class'])     # Label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel = "linear")
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,3,1,4,2,3,2,1]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)


