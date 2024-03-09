# Implementation-of-Linear-Regression-Using-Gradient-Descent
## NAME :LOSHINI.G
## REFERENCE NUMBER:212223220051
## DEPARTMENT:IT
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Start the program
2.Import numpy as np 3.Plot the points
3.IntiLiaze thhe program
4.End the program
```
   
## Program:
```
Developed by:LOSHINI.G 
RegisterNumber:  212223220051

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1, 1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1, 1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv("/content/50_Startups.csv")
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler = StandardScaler()

y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)

theta=linear_regression(X1_Scaled, Y1_Scaled)

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)

new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot (np.append(1, new_Scaled), theta)

prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)

print(f"Predicted value: {pre}")
Program to implement the linear regression using gradient descent

```

## Output:
## Given dataset:
![Screenshot 2024-03-09 092139](https://github.com/Loshini2301/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150007305/b2ffa661-2d6f-4e4f-8a67-1ac9ac8ff061)
## Predicted value:
![Screenshot 2024-03-09 091805](https://github.com/Loshini2301/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150007305/ade4a9b6-7d1c-4744-a4cf-ac764f81c5a2)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
