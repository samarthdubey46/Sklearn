import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("titanic.csv", delimiter='\t')
data.drop("Cabin", axis=1, inplace=True)
data.dropna(inplace=True)
sns.heatmap(data.isnull(), yticklabels=False)
sex = (pd.get_dummies(data['Sex'],drop_first=True))
pclass = pd.get_dummies(data['Pclass'],drop_first=True)
embark = pd.get_dummies(data['Embarked'],drop_first=True)
data = pd.concat([data,sex,pclass,embark],axis=1)
data.drop("Sex",axis=1,inplace=True)
data.drop(["PassengerId","Name","Embarked","Pclass","Ticket"],axis=1,inplace=True)
X = data.drop("Survived",axis=1)
y = data["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
L = LogisticRegression()
L.fit(X_train,y_train)
print(L.score(X_test,y_test))