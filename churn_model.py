# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



data = pd.read_excel(r'C:\Users\harsh\Documents\BEPEC Notebooks\Datasets\Churn_data.xlsx')
data.head()

data = data.drop(['CIF','CUS_DOB','CUS_Customer_Since'], axis = 1)
#data.head()
data['CUS_Gender'] = data['CUS_Gender'].ffill()
data['CUS_Month_Income'] = data['CUS_Month_Income'].ffill()

#Applying label encoder
label = LabelEncoder()
data['CUS_Gender'] = label.fit_transform(data['CUS_Gender'])
data['CUS_Marital_Status'] = label.fit_transform(data['CUS_Marital_Status'])
data['TAR_Desc'] = label.fit_transform(data['TAR_Desc'])
data['Status'] = label.fit_transform(data['Status'])

# X and y values
X = data.iloc[:,:24]
y = data['Status']

#Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state=123)
model = LogisticRegression(class_weight = {0:0.40, 1: 0.60})

#Fitting the model
model.fit(X_train, Y_train)

#Model prediction
pred = model.predict(X_test)

#Obtaining the accuracy score

print("The Accuracy is about:",accuracy_score(Y_test, pred))

confusion_matrix(Y_test, pred)
print(classification_report(Y_test, pred))

import pickle

pickle.dump(data, open("churn.pkl","wb"))

model = pickle.load(open('churn.pkl',"rb"))

#print(model.predict([[]]))

