# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
from sklearn.linear_model import LogisticRegression
import boto3

#region: 'holaMundo',
#accessKeyId: 'holaMundo',
#secretAccessKey: 'holaMundo'
s3 = boto3.resource('s3', aws_access_key_id= 'holaMundo',
         aws_secret_access_key= 'holaMundo')
s3.Object('almacenador', 'users.csv').download_file('users.csv')

data = pd.read_csv('users.csv')
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
pd.plotting.scatter_matrix(data)
plt.show()


print ("\t Menu de reportes")
print ("\t Elige una opcion de comparacion para ver una grafica")
print ("\t 1-Puntuación contra edad")
print ("\t 2-Puntuación contra carrera")
print ("\t 3-Puntuación contra semestre")
print ("\t 4-Puntuación contra genero")
print ("\t 5-Salir")
opc = int(input("\t Ingresa la opcion deseada: "))

while (opc>0 and opc<7):
    if opc==1:
        data.plot.scatter(x = 'AGE', y= 'SCORE',s=data['c']*200)
        plt.title('Prediccion de edad')
        plt.show()
        break
    elif opc==2:
        data.plot.scatter(x = 'CAREER', y= 'SCORE')
        plt.title('Prediccion de carrera')
        plt.show()
        break
    elif opc==3:
        data.plot.scatter(x = 'SEMESTER', y= 'SCORE')
        plt.title('Prediccion de semestre')
        plt.show()
        break
    elif opc==4:
        data.plot.scatter(x = 'GENDER', y= 'SCORE')
        plt.title('Prediccion de genero')
        plt.show()
        break
    else:
        print("\t Opcion no valida")
