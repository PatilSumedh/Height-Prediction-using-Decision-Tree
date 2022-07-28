#Import Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import r2_score,mean_squared_error

#Load Dataset

dataset = pd.read_csv('dataset.csv')
dataset

#Summarize Dataset

print(dataset.shape)
print(dataset.head(5))

#Segregate Dataset into Input X & Output Y

X = dataset.iloc[:, :-1].values
X

Y = dataset.iloc[:, -1].values
Y

#Splitting Dataset for Testing our Model

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

#Training Dataset using Decision Tree

clf = DecisionTreeRegressor()  
clf.fit(X_train, Y_train)

#Visualizing Graph

X_val = np.arange(min(X_train), max(X_train), 0.01) 
X_val = X_val.reshape((len(X_val), 1))

plt.scatter(X_train, Y_train, color = 'green') 
plt.plot(X_val, clf.predict(X_val), color = 'red')  
plt.title('Height prediction using DecisionTree') 
plt.xlabel('Age') 
plt.ylabel('Height') 
plt.figure()
plt.show()

#Prediction for all test data for validation

ypred = clf.predict(X_test)

mse = mean_squared_error(Y_test,ypred)
rmse=np.sqrt(mse)
print("Root Mean Square Error:",rmse)
r2score = r2_score(Y_test,ypred)
print("R2Score",r2score*100)

#Prediction Part.

age = int(input("Enter age: "))

height = [[age]]
prediction = clf.predict(height)
print(prediction)
