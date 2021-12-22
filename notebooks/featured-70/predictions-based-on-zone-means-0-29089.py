#!/usr/bin/env python
# coding: utf-8

# Predictions based off mean probabilities by zone

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read input data files
labels = pd.read_csv('../input/stage1_labels.csv')

# split the Id to create a column for zones and person ('subject')
new_list = []
for i,r in labels.iterrows():
    subject = r['Id'].split('_')[0]
    zone = r['Id'].split('_')[1][4:]
    prob = r['Probability']
    new_list.append({'Id': r['Id'], 'subject': subject, 'zone': zone, 'prob':prob})
    
df = pd.DataFrame(new_list)


# In[2]:


# get mean probabilitys by zone
zone_means = df.groupby(['zone'])['prob'].mean()
zone_means.plot.bar()

# write the csv
sample = pd.read_csv('../input/stage1_sample_submission.csv')
output = []
for i,r in sample.iterrows():
    zone = r['Id'].split('_')[1][4:]
    prob = zone_means[zone]
    output.append({'Id': r['Id'],'Probability':prob})
op_csv = pd.DataFrame(output)
op_csv.to_csv('output.csv', index=False)


# In[3]:



# write the csv
sample = pd.read_csv('../input/stage1_sample_submission.csv')
output = []
for i,r in sample.iterrows():
    zone = r['Id'].split('_')[1][4:]
    prob = zone_means[zone]
    output.append({'Id': r['Id'],'Probability':prob})
op_csv = pd.DataFrame(output)
op_csv.to_csv('output.csv', index=False)

