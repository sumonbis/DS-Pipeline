
# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np

FOLDER = "../input/google-ai-open-images-visual-relationship-track/test/challenge2018_test"
image_filenames = os.listdir(FOLDER)
fig = plt.figure(figsize=(12,12))


# In[ ]:

N_IMAGES = 10
import random
for i in range(N_IMAGES):
    path = FOLDER + "/" + image_filenames[i]
    img = mpimg.imread(path)
    plt.imshow(img[:,:,::-1], aspect='auto')
    plt.show()


# # Train Data - What is there?

# In[ ]:

train_fol = "../input/challenge-2018-train-image-relationship/challenge-2018-train-vrd.csv"

import pandas as pd
import numpy as np

df = pd.read_csv(train_fol)

# Relationships
relationship = df.RelationshipLabel.value_counts()
print(relationship)



# In[ ]:

df[df.ImageID == '00379950569d024c']


# # So, mostly relationship is 'is'

# In[ ]:

# Object 1 Label
lab_name1 = df.LabelName1.value_counts()
print(lab_name1)


# # So mostly label 1 is '/m/01mzpv' (Chair, next is Man)

# In[ ]:

lab_name2 = df.LabelName2.value_counts()
print(lab_name2)


# # Mostly Label2 is '/m/083vt'

# In[ ]:

# Combined Triplet

df['triplet'] = df['LabelName1'] + ' ' + df['RelationshipLabel'] + ' ' + df['LabelName2']
print(df.triplet.value_counts())


# In[ ]:

del df['triplet']
df.describe()


# # Lets take the MEAN and use it for all prediction

# In[ ]:

means = [0.373899,	0.624713,	0.426304,	0.785764,	0.333565,	0.666269,	0.500565,	0.807825]

sub = pd.DataFrame(columns=df.columns)
N = len(image_filenames)
sub['ImageID'] = image_filenames  
sub['LabelName1'] = ['/m/01mzpv']*N
sub['LabelName2'] = ['/m/04bcr3']*N

cols = df.columns

for i in range(3,len(cols)-1):
    col = cols[i]
    sub[col] = [means[i-3]]*N

sub['RelationshipLabel'] = ['is']*N

def get_pred(df):
    pred = '0.500000'
    pred = pred + ' ' + df['LabelName1'] + ' ' + str(df[cols[3]]) + ' ' + str(df[cols[5]]) + ' ' + str(df[cols[4]]) + ' ' + str(df[cols[6]]) + ' '+ df['LabelName2'] + ' ' +  str(df[cols[7]]) + ' ' + str(df[cols[9]]) + ' ' + str(df[cols[8]]) + ' ' + str(df[cols[10]])
    pred += ' ' + df['RelationshipLabel']
    return pred
        
sub['PredictionString'] = sub.apply(get_pred, axis=1)

final = pd.DataFrame()
def remove_jpg_ext(v):
    return v[:-4]
final['ImageId'] = sub['ImageID'].apply(remove_jpg_ext)
final['PredictionString'] = sub['PredictionString']


# In[ ]:

# sample_file = "../input/VRD_sample_submission.csv"
# import pandas as pd
# sample = pd.read_csv(sample_file)
# sample.head()


# In[ ]:




# In[ ]:

print(final.head())
final.to_csv('submission_final_chair_at_table.csv', index=False)

