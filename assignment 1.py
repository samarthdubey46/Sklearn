import pandas as pd
import sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
cancer = load_breast_cancer()
def answer_zero():
    return len(cancer['feature_names'])
print(answer_zero())
def answer_one():
    cancerdf = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']],
                            columns=np.append(cancer['feature_names'], ['target']))
    return cancerdf
def answer_two():
    cancerdf = answer_one()
    maligent  = len(cancerdf[cancerdf['target'] == 0])
    ben  = len(cancerdf[cancerdf['target'] == 1])
    t = pd.Series(data=[maligent,ben],index=['malignant','benign'])
    return t
def answer_three():
    cancerdf = answer_one()
    X = cancerdf.iloc[:, :30]
    y = cancerdf.iloc[:, 30]
    return X, y


def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors=1)
    s = knn.fit(X_train, y_train)
    return s


def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    predict = answer_five()
    label = predict.predict(means)
    return label

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.predict(X_test)


def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.score(X_test, y_test)
print(answer_eight())