# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:05:42 2021

@author: MR. Hacker
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import pickle as pkl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv(r"C:\Users\user\Credit\creditcard.csv")
data
data.head()
data.tail()
data.info()
#checking the number of the missing values in each column
data.isnull().sum()
#distribution of the legist transacation and the fraud tracnsation
data['Class'].value_counts()
#Sepreating data for the data analysis
legist=data[data.Class==0]
fraud=data[data.Class==1]
legist.shape
fraud.shape
#statstical measure of data
legist.Amount.describe()
fraud.Amount.describe()
#compare the value of the both transaction
data.groupby('Class').mean()
#Build the sample of the dataset containing similar distribution of normal and fraud transaction
legist_sample = legist.sample(n=492)
#concatenate two dataframes
new_data=pd.concat([legist_sample,fraud],axis=0)
new_data.head()
new_data.tail()
new_data["Class"].value_counts()
new_data.groupby("Class").mean()
#Split the data into feature and traget
X=new_data.drop(columns="Class",axis=1)
Y=new_data["Class"]
print(X)
print(Y)
#split the data into the training data and the testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)
print(X.shape,X_train.shape,X_test.shape)
#Logistic Regression
model = LogisticRegression()
#Training the logistic regression model with a training data
model_learner=model.fit(X_train,Y_train)
#Model Evalution
#Accuracy Score
#Accuracy in the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy of training data: ",training_data_accuracy)
#Accuracy on the tewst data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the test data: ',test_data_accuracy)
filename='credit.pkl'
with open('credit.pkl','wb')as fh:
    pkl.dump(model_learner,fh)
