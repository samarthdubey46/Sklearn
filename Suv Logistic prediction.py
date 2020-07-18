import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("suv_data.csv")
# print(data.head())
X = data.iloc[:,2:4]
y = data['Purchased']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
m = MinMaxScaler()
X_scaled = m.fit_transform(X_train)
X_test_scaled = m.fit_transform(X_test)
L = LogisticRegression(penalty='l2')
L.fit(X_scaled,y_train)
x = [[39,89000]]
z = (L.predict(x))
print(y_test[z[0]])
