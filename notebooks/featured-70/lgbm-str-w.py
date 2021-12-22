
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import lightgbm as lgb


# In[2]:


t_res = pd.read_csv('../input/wdatafiles/WNCAATourneyCompactResults.csv')
t_ds = pd.read_csv('../input/wdatafiles/WNCAATourneySeeds.csv')
sub = pd.read_csv('../input/WSampleSubmissionStage1.csv')


# In[3]:


t_ds['seed_int'] = t_ds.Seed.apply(lambda a : int(a[1:3]))


# In[4]:


drop_lbls = ['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT']
t_ds.drop(labels=['Seed'], inplace=True, axis=1)
t_res.drop(labels=drop_lbls, inplace=True, axis=1)


# In[5]:


ren1 = {'TeamID':'WTeamID', 'seed_int':'WS'}
ren2 = {'TeamID':'LTeamID', 'seed_int':'LS'}


# In[6]:


df1 = pd.merge(left=t_res, right=t_ds.rename(columns=ren1), how='left', on=['Season', 'WTeamID'])
df2 = pd.merge(left=df1, right=t_ds.rename(columns=ren2), on=['Season', 'LTeamID'])

df_w = pd.DataFrame()
df_w['dff'] = df2.WS - df2.LS
df_w['rsl'] = 1

df_l = pd.DataFrame()
df_l['dff'] = -df_w['dff']
df_l['rsl'] = 0

df_prd = pd.concat((df_w, df_l))


# In[7]:


X = df_prd.dff.values.reshape(-1,1)
y = df_prd.rsl.values


# In[8]:


X_test = np.zeros(shape=(len(sub), 1))


# In[9]:


for ind, row in sub.iterrows():
    yr, o, t = [int(x) for x in row.ID.split('_')]  
    X_test[ind, 0] = t_ds[(t_ds.TeamID == o) & (t_ds.Season == yr)].seed_int.values[0] - t_ds[(t_ds.TeamID == t) & (t_ds.Season == yr)].seed_int.values[0]


# In[10]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)


# In[11]:


params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": 0.01,
          "num_leaves": 31,
          "min_data_in_leaf": 10,
          "min_child_samples": 10,
          }

dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
bst = lgb.train(params, dtrain, 1000, valid_sets=dvalid, verbose_eval=200,
    early_stopping_rounds=50)


# In[12]:


test_pred = bst.predict(
    X_test, num_iteration=bst.best_iteration)


# In[13]:


sub.Pred = test_pred   
sub.to_csv('sub.csv', index=False)

