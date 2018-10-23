import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('users.csv')
print(data.columns.values)
#pd.scatter_matrix(data)
data.plot.scatter(x = 'AGE', y= 'SCORE')
plt.title('Titulo')
#data.plot.bar(x = 'AGE', y= 'SCORE')
plt.show()

x = np.asanyarray(data[['AGE','CAREER','SEMESTER','GENDER']])
y = np.asanyarray(data[['SCORE']]).ravel()

poly = pr.PolynomialFeatures(degree=2)
x = poly.fit_transform(x)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y)
print(x.shape)
model = lm.LogisticRegression()
model.fit(xtrain, ytrain)

print('train: ',model.score(xtrain, ytrain))
print('test', model.score(xtest, ytest))
