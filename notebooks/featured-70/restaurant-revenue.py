#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
trainData = trainData.drop('Id', axis=1)
testData = testData.drop('Id', axis=1)


# In[2]:


trainData['Open Date'] = pd.to_datetime(trainData['Open Date'], format='%m/%d/%Y')   
testData['Open Date'] = pd.to_datetime(testData['Open Date'], format='%m/%d/%Y')

trainData['OpenDays']=""
testData['OpenDays']=""

dateLastTrain = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(trainData)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y')  
dateLastTest = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(testData)]) })
dateLastTest['Date'] = pd.to_datetime(dateLastTest['Date'], format='%m/%d/%Y')  

trainData['OpenDays'] = dateLastTrain['Date'] - trainData['Open Date']
testData['OpenDays'] = dateLastTest['Date'] - testData['Open Date']

trainData['OpenDays'] = trainData['OpenDays'].astype('timedelta64[D]').astype(int)
testData['OpenDays'] = testData['OpenDays'].astype('timedelta64[D]').astype(int)

trainData = trainData.drop('Open Date', axis=1)
testData = testData.drop('Open Date', axis=1)


# In[3]:


cityPerc = trainData[["City Group", "revenue"]].groupby(['City Group'],as_index=False).mean()
#sns.barplot(x='City Group', y='revenue', data=cityPerc)

citygroupDummy = pd.get_dummies(trainData['City Group'])
trainData = trainData.join(citygroupDummy)

citygroupDummyTest = pd.get_dummies(testData['City Group'])
testData = testData.join(citygroupDummyTest)

trainData = trainData.drop('City Group', axis=1)
testData = testData.drop('City Group', axis=1)


# In[4]:


#Regression on everything
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

import numpy
xTrain = pd.DataFrame({'OpenDays':trainData['OpenDays'].apply(numpy.log),
                      'Big Cities':trainData['Big Cities'], 'Other':trainData['Other'],
                      'P2':trainData['P2'], 'P8':trainData['P8'], 'P22':trainData['P22'],
                      'P24':trainData['P24'], 'P28':trainData['P28'], 'P26':trainData['P26']})
#xTrain = trainData.drop(['revenue'], axis=1)
#xTrain['OpenDays'] = xTrain['OpenDays'].apply(numpy.log)
yTrain = trainData['revenue'].apply(numpy.log)
xTest = pd.DataFrame({'OpenDays':testData['OpenDays'].apply(numpy.log),
                      'Big Cities':testData['Big Cities'], 'Other':testData['Other'],
                     'P2':testData['P2'], 'P8':testData['P8'], 'P22':testData['P22'],
                      'P24':testData['P24'], 'P28':testData['P28'], 'P26':testData['P26']})

from sklearn import linear_model

cls = RandomForestRegressor(n_estimators=150)
cls.fit(xTrain, yTrain)
pred = cls.predict(xTest)
pred = numpy.exp(pred)
cls.score(xTrain, yTrain)


# In[5]:


pred = cls.predict(xTest)
pred = numpy.exp(pred)


# In[6]:


pred


# In[7]:


pred2 = []
for i in range(len(pred)):
    if pred[i] != float('Inf'):
        pred2.append(pred[i])

m = sum(pred2) / float(len(pred2))

for i in range(len(pred)):
    if pred[i] == float('Inf'):
        print("haha")
        pred[i] = m


# In[8]:


testData = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({
        "Id": testData["Id"],
        "Prediction": pred
    })
submission.to_csv('RandomForestSimple.csv',header=True, index=False)

