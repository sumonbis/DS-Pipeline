#!/usr/bin/env python
# coding: utf-8

# Forked from https://www.kaggle.com/ilya16/lstm-models?scriptVersionId=10420679 with refactoring, simplification and some changes to the model
# 
# Data Preprocessing and Deep LSTM model are inspired by the top solution described here: 
# http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/

# In[1]:


import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))


# In[2]:


N_FEATURES = 22

# taken from http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
THRESHOLD = 73 


# # Data preprocessing

# ## Training set

# In[3]:


train_df = pd.read_csv("../input/train.csv")


# In[4]:


# to reduce memory consumption
train_df[train_df.columns[1:]] = train_df[train_df.columns[1:]].astype(np.float32)


# In[5]:


train_df.shape


# Remove ids with NaNs in `Ref` column for each observation (obeservations, where we have no data from radar)

# In[6]:


good_ids = set(train_df.loc[train_df['Ref'].notna(), 'Id'])
train_df = train_df[train_df['Id'].isin(good_ids)]
train_df.shape


# Replace NaN values with zeros

# In[7]:


train_df.fillna(0.0, inplace=True)
train_df.reset_index(drop=True, inplace=True)
train_df.head()


# In[8]:


train_df.shape


# Define and exclude outliers from training set

# In[9]:


train_df = train_df[train_df['Expected'] < THRESHOLD]


# In[10]:


train_df.shape


# ### Grouping and padding into sequences

# In[11]:


train_groups = train_df.groupby("Id")
train_size = len(train_groups)


# In[12]:


MAX_SEQ_LEN = train_groups.size().max()
MAX_SEQ_LEN


# In[13]:


X_train = np.zeros((train_size, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
y_train = np.zeros(train_size, dtype=np.float32)

i = 0
for _, group in train_groups:
    X = group.values
    seq_len = X.shape[0]
    X_train[i,:seq_len,:] = X[:,1:23]
    y_train[i] = X[0,23]
    i += 1
    del X
    
del train_groups
X_train.shape, y_train.shape


# ## Test set

# In[14]:


test_df = pd.read_csv("../input/test.csv")
test_df[test_df.columns[1:]] = test_df[test_df.columns[1:]].astype(np.float32)
test_ids = test_df['Id'].unique()

# Convert all NaNs to zero
test_df = test_df.fillna(0.0)
test_df = test_df.reset_index(drop=True)


# In[15]:


test_groups = test_df.groupby("Id")
test_size = len(test_groups)

X_test = np.zeros((test_size, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)

i = 0
for _, group in test_groups:
    X = group.values
    seq_len = X.shape[0]
    X_test[i,:seq_len,:] = X[:,1:23]
    i += 1
    del X
    
del test_groups
X_test.shape


# # Models

# In[16]:


from keras.layers import (
    Input,
    Dense,
    LSTM,
    AveragePooling1D,
    TimeDistributed,
    Flatten,
    Bidirectional,
    Dropout
)
from keras.models import Model


# In[17]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=0.01)


# In[18]:


BATCH_SIZE = 1024
N_EPOCHS = 30


# ## Deep model
# 
# Deep NN inspired by the top solution

# In[19]:


def get_model_deep(shape=(19, 22)):
    inp = Input(shape)
    x = Dense(16)(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = TimeDistributed(Dense(64))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = TimeDistributed(Dense(1))(x)
    x = AveragePooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)

    model = Model(inp, x)
    return model


# In[20]:


model = get_model_deep((19,22))
model.compile(optimizer='adam', loss='mae',)
model.summary()


# In[21]:


model.fit(X_train, y_train, 
            batch_size=BATCH_SIZE, epochs=N_EPOCHS, 
            validation_split=0.2, callbacks=[early_stopping, reduce_lr])


# In[22]:


y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
submission = pd.DataFrame({'Id': test_ids, 'Expected': y_pred.reshape(-1)})
submission.to_csv('submission.csv', index=False)


# In[23]:




