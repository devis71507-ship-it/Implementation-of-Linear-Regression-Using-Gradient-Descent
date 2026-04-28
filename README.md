# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset.
   
2.Preprocess data (handle missing values and split features/targets).

3.Scale features using StandardScaler.

4.Train the model using SGDRegressor.

5.Predict house price and number of occupants and evaluate the model.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DEVI S
RegisterNumber:  212225100008

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_erro
X = np.array([
    [1200, 3, 1],
    [1500, 4, 2],
    [800, 2, 1],
    [2000, 5, 3],
    [1700, 4, 2],
    [1000, 2, 1],
    [2200, 5, 3],
    [1300, 3, 2]
])
# Targets: [Price (in lakhs), Number of Occupants]
y = np.array([
    [50, 4],
    [65, 5],
    [35, 3],
    [90, 7],
    [70, 6],
    [40, 3],
    [100, 8],
    [55, 4]
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant')
multi_regressor = MultiOutputRegressor(sgd)
multi_regressor.fit(X_train, y_train)
y_pred = multi_regressor.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted.round(2)}")
```

## Output:

<img width="423" height="150" alt="Screenshot 2026-04-28 203537" src="https://github.com/user-attachments/assets/9d328a5e-05c8-490e-b330-68d9325b3c57" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
