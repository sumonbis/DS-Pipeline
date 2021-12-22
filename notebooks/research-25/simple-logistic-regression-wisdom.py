#!/usr/bin/env python
# coding: utf-8

# # Import library

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import datetime as dt

get_ipython().run_line_magic('matplotlib', 'inline')
seed = 129


# # Import Dataset

# In[2]:


path = '../input/'
#path = ''
train = pd.read_csv(path+'train_v2.csv',dtype={'is_churn':np.int8})
test = pd.read_csv(path+'sample_submission_v2.csv',dtype={'is_churn':np.int8})
members = pd.read_csv(path+'members_v3.csv',parse_dates=['registration_init_time'],dtype={'city':np.int8,'bd':np.int8,
                                                                                         'registered_via':np.int8})
transactions = pd.read_csv(path+'transactions_v2.csv',parse_dates=['transaction_date','membership_expire_date'],
                          dtype={'payment_method_id':np.int8,'payment_plan_days':np.int8,'plan_list_price':np.int8,
                                'actual_amount_paid':np.int8,'is_auto_renew':np.int8,'is_cancel':np.int8})

user_log = pd.read_csv(path+'user_logs_v2.csv',parse_dates=['date'],dtype={'num_25':np.int16,'num_50':np.int16,
                                    'num_75':np.int16,'num_985':np.int16,'num_100':np.int16,'num_unq':np.int16})


# # Explore data set

# In[3]:


print('Number of rows  & columns',train.shape)
train.head()


# In[4]:


print('Number of rows  & columns',test.shape)
test.head()


# In[5]:


print('Number of rows  & columns',members.shape)
members.head()


# In[6]:


print('Number of rows & columns',transactions.shape)
transactions.head()


# In[7]:


print('Number of rows & columns',user_log.shape)
user_log.head()


# In[8]:


print('\nTrain:',train.describe().T)
print('\nTest:',test.describe().T)
print('\nMembers:',members.describe().T)
print('\nTransactions:',transactions.describe().T)
print('\nUser log:',user_log.describe().T)


# # Merge data set

# In[9]:


train = pd.merge(train,members,on='msno',how='left')
test = pd.merge(test,members,on='msno',how='left')
train = pd.merge(train,transactions,how='left',on='msno',left_index=True, right_index=True)
test = pd.merge(test,transactions,how='left',on='msno',left_index=True, right_index=True,)
train = pd.merge(train,user_log,how='left',on='msno',left_index=True, right_index=True)
test = pd.merge(test,user_log,how='left',on='msno',left_index=True, right_index=True)

del members,transactions,user_log
print('Number of rows & columns',train.shape)
print('Number of rows & columns',test.shape)


# # Date feature

# In[10]:


train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].describe()


# In[11]:


train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].isnull().sum()


# In[12]:


train['registration_init_time'] = train['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))
test['registration_init_time'] = test['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))


# In[13]:


def date_feature(df):
    
    col = ['registration_init_time' ,'transaction_date','membership_expire_date','date']
    var = ['reg','trans','mem_exp','user_']
    #df['duration'] = (df[col[1]] - df[col[0]]).dt.days 
    
    for i ,j in zip(col,var):
        df[j+'_day'] = df[i].dt.day.astype('uint8')
        df[j+'_weekday'] = df[i].dt.weekday.astype('uint8')        
        df[j+'_month'] = df[i].dt.month.astype('uint8') 
        df[j+'_year'] =df[i].dt.year.astype('uint16') 

date_feature(train)
date_feature(test)


# # Data analysis 

# In[14]:


train.columns


# # Missing value

# In[15]:


train.isnull().sum()


# In[16]:


train.info()


# In[17]:


col = [ 'city', 'bd', 'gender', 'registered_via']
def missing(df,columns):
    col = columns
    for i in col:
        df[i].fillna(df[i].mode()[0],inplace=True)

missing(train,col)
missing(test,col)


# In[18]:


def unique_value(df):
    col = df.columns
    for i in col:
        print('Number of unique value in {} is {}'.format(i,df[i].nunique()))

unique_value(train)


# # is_churn

# In[19]:


plt.figure(figsize=(8,6))
sns.set_style('ticks')
sns.countplot(train['is_churn'],palette='summer')
plt.xlabel('The subscription within 30 days of expiration is True/False')


# Imbalanced data set
# 
# msno: user id
# 
# is_churn: This is the target variable. Churn is defined as whether the user did not continue the subscription within 30 days of expiration. 
# is_churn = 1 means churn,is_churn = 0 means renewal.

# ## Univariate analysis

# In[20]:


print(train['city'].unique())
fig,ax = plt.subplots(2,2,figsize=(16,8))
ax1,ax2,ax3,ax4 = ax.flatten()

sns.set(style="ticks")
sns.countplot(train['city'],palette='summer',ax=ax1)
#ax1.set_yscale('log')

ax1.set_xlabel('City')
#ax1.set_xticks(rotation=45)

sns.countplot(x='gender',data = train,palette='winter',ax=ax2)
#ax2.set_yscale('log')
ax2.set_xlabel('Gender')

sns.countplot(x='registered_via',data=train,palette='winter',ax=ax3)
#ax3.set_yscale('')
ax3.set_xlabel('Register via')

sns.countplot(x='payment_method_id',data= train,palette='winter',ax=ax4)
ax4.set_xlabel('Payment_method_id')


# # bd  (birth day)

# In[21]:


print(train['bd'].describe())


# In[22]:


fig,ax = plt.subplots(1,2,figsize=(16,8))
ax1,ax2 = ax.flatten()
sns.set_style('ticks')
sns.distplot(train['bd'].fillna(train['bd'].mode()[0]),bins=100,color='r',ax=ax1)
plt.title('Distribution of birth day')


# In[23]:


plt.figure(figsize=(14,6))
sns.distplot(train.loc[train['bd'].value_counts()]['bd'].fillna(0),bins=50,color='b')


# # Gender

# In[24]:


print(pd.crosstab(train['is_churn'],train['gender']))


# # registration_init_time

# In[25]:


regi = train.groupby('registration_init_time').count()['is_churn']
plt.subplot(211)
plt.plot(regi,color='b',label='count')
plt.legend(loc='center')
regi = train.groupby('registration_init_time').mean()['is_churn']
plt.subplot(212)
plt.plot(regi,color='r',label='mean')
plt.legend(loc='center')
plt.tight_layout()


# In[26]:


regi = train.groupby('registration_init_time').mean()['is_churn']
plt.figure(figsize=(14,6))
sns.distplot(regi,bins=100,color='r')


# # registration

# In[27]:


fig,ax = plt.subplots(2,2,figsize=(16,8))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.countplot(train['reg_day'],palette='Set2',ax=ax1)
sns.countplot(data=train,x='reg_month',palette='Set1',ax=ax2)
sns.countplot(data=train,x='reg_year',palette='magma',ax=ax3)


# In[28]:


cor = train.corr()
plt.figure(figsize=(16,12))
sns.heatmap(cor,cmap='Set1',annot=False)
plt.xticks(rotation=45);


# # Encoder

# In[29]:


le = LabelEncoder()
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])


# # One Hot Encoding

# In[30]:


def OHE(df):
    #col = df.select_dtypes(include=['category']).columns
    col = ['city','gender','registered_via']
    print('Categorical columns in dataset',col)
    
    c2,c3 = [],{}
    for c in col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df,columns=c2,drop_first=True,prefix=c3)
    print(df.shape)
    return df
train1 = OHE(train)
test1 = OHE(test)


# In[31]:


train1.columns


# # Split data set

# In[32]:


unwanted = ['msno','is_churn','registration_init_time','transaction_date','membership_expire_date','date']

X = train1.drop(unwanted,axis=1)
y = train1['is_churn'].astype('category')
x_test = test1.drop(unwanted,axis=1)


# ## Hyper parameter tuning

# log_reg = LogisticRegression(class_weight='balanced')
# param = {'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,3]}
# rs_cv = RandomizedSearchCV(estimator=log_reg,param_distributions=param,random_state=seed)
# rs_cv.fit(X,y)
# print('Best parameter :{} Best score :{}'.format(rs_cv.best_params_,rs_cv.best_score_))

# # Logistic regression model with Stratified KFold split

# #
# 
# kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
# pred_test_full =0
# cv_score =[]
# i=1
# for train_index,test_index in kf.split(X,y):
#     print('{} of KFold {}'.format(i,kf.n_splits))
#     xtr,xvl = X.loc[train_index],X.loc[test_index]
#     ytr,yvl = y.loc[train_index],y.loc[test_index]    
#     #model
#     lr = LogisticRegression(class_weight='balanced',C=1)
#     lr.fit(xtr,ytr)
#     score = lr.score(xvl,yvl)
#     print('ROC AUC score:',score)
#     cv_score.append(score)    
#     pred_test = lr.predict_proba(x_test)[:,1]
#     pred_test_full +=pred_test
#     i+=1

# In[33]:


lr = LogisticRegression(class_weight='balanced',C=1)
lr.fit(X,y)
y_pred = lr.predict_proba(x_test)[:,1]
lr.score(X,y)


# # Model validation

# print(cv_score)
# print('\nMean accuracy',np.mean(cv_score))
# confusion_matrix(yvl,lr.predict(xvl))

# ## Reciever Operating Charactaristics

# In[34]:


y_proba = lr.predict_proba(X)[:,1]
fpr,tpr,th = roc_curve(y,y_proba)

plt.figure(figsize=(14,6))
plt.plot(fpr,tpr,color='r')
plt.plot([0,1],[0,1],color='b')
plt.title('Reciever operating Charactaristics')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')


# # Predict for unseen data set

# In[35]:


#y_pred = pred_test_full/5
submit = pd.DataFrame({'msno':test['msno'],'is_churn':y_pred})
submit.to_csv('kk_pred.csv',index=False)
#submit.to_csv('kk_pred.csv.gz',index=False,compression='gzip')


# # Thank you for visiting
