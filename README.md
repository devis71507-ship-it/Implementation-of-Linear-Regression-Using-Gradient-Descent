# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect training data (city features and profit values).

2.Initialize parameters (θ0, θ1) and learning rate (α).

3.Compute predicted profit using hypothesis function.

4.Update parameters using gradient descent to reduce error.

5.Repeat until convergence and use final model for prediction.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DEVI S
RegisterNumber:  212225100008

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression (X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors= (predictions-y).reshape(-1,1)
    theta -learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv(r"C:\Users\acer\Downloads\50_Startups.csv")
data.head(11)
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print("X =",X)
print("X1_Scaled =",X1_Scaled)
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform (prediction)
print("prediction =",prediction)
print(f"Predicted value: {pre}")
```

## Output:

<img width="644" height="461" alt="image" src="https://github.com/user-attachments/assets/67cbfac8-a146-4d1d-92a2-13e2f6a72336" />
<img width="507" height="824" alt="image" src="https://github.com/user-attachments/assets/44245d6b-23fe-47da-bbcd-543091e5ad1a" />
<img width="583" height="792" alt="image" src="https://github.com/user-attachments/assets/b4e440f5-956d-4034-8021-8895f8ef8dc1" />
<img width="485" height="252" alt="image" src="https://github.com/user-attachments/assets/f78898c0-fd3e-4c60-9301-92761c7f6144" />
<img width="719" height="44" alt="image" src="https://github.com/user-attachments/assets/949da81c-e3fc-42a9-a14b-1ead1b44158d" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
