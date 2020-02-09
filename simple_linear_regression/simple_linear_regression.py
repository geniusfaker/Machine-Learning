import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('C:\\Users\geniu\Downloads\Compressed\simple-Linear-Regression-master/Salary_Data.csv')

#split dataset into dependent and independent variables

X=df.iloc[:,:-1].values
y=df.iloc[:, 1].values

#dividing the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Implement classifiers based on Simple linear Regression

from sklearn.linear_model import LinearRegression
simple_linear_regressor=LinearRegression()
simple_linear_regressor.fit(X_train,y_train)

y_pred =simple_linear_regressor.predict(X_test)

#implement graph for simple linear regression
plt.scatter(X_train, y_train, color='red')
plt.plot(X_test,y_pred)
plt.show()
