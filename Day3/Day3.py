import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
#print(dataset)
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,4].values
#print(X)
#print(Y)

getColumn = pd.get_dummies(dataset['State'])
#print(getColumn)
concat = pd.concat([X,getColumn], axis=1)
X = concat.drop(['State'], axis=1)
#print(X)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
#print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print(X_test)
y_pred = regressor.predict(X_test)
print(y_pred)

