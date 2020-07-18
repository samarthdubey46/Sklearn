import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
fruits = pd.read_table("train.txt")
lookup = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
X = fruits[['mass','width','height']]
y = fruits[['fruit_label']]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
z = knn.predict([[40,0.5,9.5]])
print(lookup[z[0]])
# linearg = LinearRegression().fit(X_train,y_train)
# print(f"b : {linearg.intercept_}" + "\n" + f"w {linearg.coef_}")