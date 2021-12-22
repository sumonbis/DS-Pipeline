#!/usr/bin/env python
# coding: utf-8

# ## Unsupervised categorical encodings
# 
# We will start with "unsupervised" categorical encodings. By “unsupervised” here I mean we are not going to use information about the target in any way. 
# 
# One of the advantages of this approach - you can use all data you have, for example in case of Kaggle competition you can use both files, `train.csv` and `test.csv`. 
# 
# In first cell we load all data we need.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from catboost.datasets import amazon
train, test = amazon()
print(train.shape, test.shape)
target = "ACTION"
col4train = [x for x in train.columns if x not in [target, "ROLE_TITLE"]]
y = train[target].values


# Here some helper functions.

# In[2]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

# returns model instance
def get_model(): 
    params = {
        "n_estimators":300, 
        "n_jobs": 3,
        "random_state":5436,
    }
    return ExtraTreesClassifier(**params)

# validate model on given dataset and report CV score
def validate_model(model, data):
    skf = StratifiedKFold(n_splits=5, random_state = 4141, shuffle = True)
    stats = cross_validate(
        model, data[0], data[1], 
        groups=None, scoring='roc_auc', 
        cv=skf, n_jobs=None, return_train_score = True
    )
    stats = pd.DataFrame(stats)
    return stats.describe().transpose()

# transforms given train and test datasets using provided function, 
# function parameters can be passed as a dict
def transform_dataset(train, test, func, func_params = {}):
    dataset = pd.concat([train, test], ignore_index = True)
    dataset = func(dataset, **func_params)
    if isinstance(dataset, pd.DataFrame):
        new_train = dataset.iloc[:train.shape[0],:].reset_index(drop = True)
        new_test =  dataset.iloc[train.shape[0]:,:].reset_index(drop = True)
    else:
        new_train = dataset[:train.shape[0]]
        new_test =  dataset[train.shape[0]:]
    return new_train, new_test


# ## Label Encoding
# 
# First way to encode your categorical feature is slightly counterintuitive, you just randomly assign unique integer value for each category level. 
# 
# And because it is randomly assigned why limit ourselves to just one column? Let do arbitrary amount of integers assigned. 
# 
# **Advantages** - because values are randomly assigned, they are unbiased. 
# 
# **Disadvantages** - hard to explain to business users why did you do that.

# In[3]:


MJTCP = 32292 #Michael Jordan total career points
#for each column in dataset creates N column with random integers
def assign_rnd_integer(dataset, number_of_times = 5, seed = MJTCP):
    new_dataset = pd.DataFrame()
    np.random.seed(seed)
    for c in dataset.columns:
        for i in range(number_of_times):
            col_name = c+"_"+str(i)
            unique_vals = dataset[c].unique()
            labels = np.array(list(range(len(unique_vals))))
            np.random.shuffle(labels)
            mapping = pd.DataFrame({c: unique_vals, col_name: labels})
            new_dataset[col_name] = (dataset[[c]]
                                     .merge(mapping, on = c, how = 'left')[col_name]
                                    ).values
    return new_dataset


# In[4]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    assign_rnd_integer, {"number_of_times":5}
)
print(new_train.shape, new_test.shape)
new_train.head(5)


# As you can see for each of 8 columns we've made 5 columns with assigned random integers, which gives us 40 columns in total.

# In[5]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# Ok, let's run the same transformation but with 1 and 10 columns instead of 5.

# In[6]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    assign_rnd_integer, {"number_of_times":1}
)
print(new_train.shape, new_test.shape)
validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# In[7]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    assign_rnd_integer, {"number_of_times":10}
)
print(new_train.shape, new_test.shape)
validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# As you can see for 10 columns we've got AUC score of 0.8663, while for 1 column score was only 0.7843.

# ## One-hot encoding
# 
# Now let get back to one-hot encoding and check how it works for tree-based model.
# 
# **Advantages** 
# * easy to implement
# * easy to understand
# * for tree-based models like RF helps to build slightly more robust models
# 
# **Disadvantages**
# * Working with very sparse matrices can be hard
# * Tree-based models can drop in performance on this type of encoding

# In[8]:


from sklearn.preprocessing import OneHotEncoder
# transforms given dataset to OHE representation
def one_hot(dataset):
    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
    return ohe.fit_transform(dataset.values)


# In[9]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    one_hot)
print(new_train.shape, new_test.shape)


# In[10]:


#Warning!!! Long run, better skip it.
validate_model(
    model = get_model(), 
    data = [new_train, y]
)


# As you can see results are good. But running time is bigger too. Unfortunately, one-hot encoding creates a very sparse data, and all tree-based models have trouble with that type of data which could lead to lower performance scores or longer time to train (not true for libraries such as XGBoost or LightGBM, they can handle sparse data pretty good).

# ## SVD Encoding
# 
# In this embedding we will try to incorporate information about data interaction.
# 
# Take 2 columns A and B, and present them as 2D matrix where rows are unique values of column A and columns are unique values of column B, in a cell of this matrix you'll put a number of times these 2 unique values can be found in dataset.
# 
# You also can apply IDF (inverse document transform) in order to assign bigger values rare pairs. 
# 
# After that you apply dimensionality reduction technique to this 2D matrix (Truncated SVD in our case) and reduce this matrix to a vector. That vector will be embedding of column A based on column B. 
# 
# Repeat this process for all columns in your dataset.
# 
# **Advantages** 
# * could provide rich and multidimensional representations
# 
# **Disadvantages**
# * Not really easy to understand what it's actually represents

# In[11]:


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def extract_col_interaction(dataset, col1, col2, tfidf = True):
    data = dataset.groupby([col1])[col2].agg(lambda x: " ".join(list([str(y) for y in x])))
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))
    else:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "))
    
    data_X = vectorizer.fit_transform(data)
    dim_red = TruncatedSVD(n_components=1, random_state = 5115)
    data_X = dim_red.fit_transform(data_X)
    
    result = pd.DataFrame()
    result[col1] = data.index.values
    result[col1+"_{}_svd".format(col2)] = data_X.ravel()
    return result

import itertools
def get_col_interactions_svd(dataset, tfidf = True):
    new_dataset = pd.DataFrame()
    for col1,col2 in itertools.permutations(dataset.columns, 2):
        data = extract_col_interaction(dataset, col1,col2, tfidf)
        col_name = [x for x in data.columns if "svd" in x][0]
        new_dataset[col_name] = dataset[[col1]].merge(data, on = col1, how = 'left')[col_name]
    return new_dataset


# In[12]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    get_col_interactions_svd
)
print(new_train.shape, new_test.shape)
new_train.head(5)


# In[13]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# And results are not very satisfying - AUC is 0.8616.

# ## Frequency encoding
# 
# The final way to encode our categorical features is frequency encoding - just count how many time this unique value is presented in your data.
# 
# 
# **Advantages** 
# * easy to implement
# * easy to understand
# 
# **Disadvantages**
# * Can't be used in case if you categorical values are balanced

# In[14]:


def get_freq_encoding(dataset):
    new_dataset = pd.DataFrame()
    for c in dataset.columns:
        data = dataset.groupby([c]).size().reset_index()
        new_dataset[c+"_freq"] = dataset[[c]].merge(data, on = c, how = "left")[0]
    return new_dataset


# In[15]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    get_freq_encoding
)
print(new_train.shape, new_test.shape)
new_train.head(5)


# In[16]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# So far the worst results - AUC is 0.8209. Linear model on OHE gives us AUC of 0.86

# Of course all this encodings can be combined.

# In[17]:


new_train1, new_test1 = transform_dataset(
    train[col4train], test[col4train], get_freq_encoding
)
new_train2, new_test2 = transform_dataset(
    train[col4train], test[col4train], get_col_interactions_svd
)
new_train3, new_test3 = transform_dataset(
    train[col4train], test[col4train], 
    assign_rnd_integer, {"number_of_times":10}
)

new_train = pd.concat([new_train1, new_train2, new_train3], axis = 1)
new_test = pd.concat([new_test1, new_test2, new_test3], axis = 1)
print(new_train.shape, new_test.shape)


# In[18]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# Nice, AUC is 0.8782. Let's see how we perform on leaderboard.

# In[19]:


model = get_model()
model.fit(new_train.values, y)
predictions = model.predict_proba(new_test)[:,1]

submit = pd.DataFrame()
submit["Id"] = test["id"]
submit["ACTION"] = predictions

submit.to_csv("submission.csv", index = False)


# In[20]:




