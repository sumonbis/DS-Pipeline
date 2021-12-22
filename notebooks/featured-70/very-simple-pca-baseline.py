#!/usr/bin/env python
# coding: utf-8

# ## Hi, everyone! 
# ## Here's a simple baseline based on PCA features extracted from images. There are many things you can to do on top of that, e.g. tuning number of principal components, varying size of training sample for PCA, splitting image by channels etc.
# #### Credits for the public starter to @inversion

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
from pathlib import Path
import multiprocessing as mp

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.data import imread
from sklearn.ensemble import RandomForestClassifier
import time
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.restoration import  estimate_sigma

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


input_path = Path('../input')
train_path = input_path / 'train'
test_path = input_path / 'test'


# In[3]:


cameras = os.listdir(train_path)

train_images = []
for camera in cameras:
    for fname in sorted(os.listdir(train_path / camera)):
        train_images.append((camera, fname))

train = pd.DataFrame(train_images, columns=['camera', 'fname'])
print(train.shape)


# In[4]:


test_images = []
for fname in sorted(os.listdir(test_path)):
    test_images.append(fname)

test = pd.DataFrame(test_images, columns=['fname'])
print(test.shape)


# ### We define a center crop and will compute PCA on cropped train images. 

# In[5]:


def get_center_crop(img, d=250): # delta
    cy = img.shape[0] // 2
    cx = img.shape[1] // 2
    return img[cy - d:cy + d, cx - d:cx + d]


# ### We define PCA with5 principal components, which we later input to Random Forest. Indeed, it's _far_ not the best way to compute PCA only on train non-altered images, but let's give it a try!

# In[6]:


n_components = 5
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True)

# Get some training data for PCA
random_images = train.sample(300)
random_images[:5]


# In[7]:


img_set_reds = []
for i, r in random_images.iterrows():
    # If you uncomment last part, you can extract features only over a certain channel
    x = get_center_crop(cv2.imread("../input/train/" + r['camera'] + '/' + r['fname']))#[:,:,0] 
    img_set_reds.append(np.ravel(x)) # PCA takes instances as flatten vectors, not 2-d array
img_set_reds = np.asarray(img_set_reds)
print(img_set_reds.shape)
print([img_set_reds[i].shape for i in range(10)])
pf = pca.fit(np.asarray(img_set_reds))


# In[8]:


def get_pca_features(img):
    img = np.ravel(img).reshape(1, -1)
    return pf.transform(img)


# In[9]:


t = get_pca_features(get_center_crop(
        cv2.imread("../input/train/" + r['camera'] + '/' + r['fname']))
)
t[0]


# In[10]:


def color_stats(q, iolock):
    while True:
        img_path = q.get()
        if img_path is None:
            break
        if type(img_path) is tuple:
            img = imread(train_path / img_path[0] / img_path[1])
            key = img_path[1]
        else:
            img = imread(test_path / img_path)
            key = img_path         
        # Some images read return info in a 2nd dim. We only want the first dim.
        if img.shape == (2,):
            img = img[0]
        # crop to center as in test    
        img = get_center_crop(img)
        pca_feats = get_pca_features(img)
        # Estimate the average noise standard deviation across color channels.
        # average_sigmas = True if you want to average across channels
        rgb_sigma_est = estimate_sigma(img, multichannel=True, average_sigmas=False)
        color_info[key] = ( pca_feats[0][0],pca_feats[0][1],pca_feats[0][2],
          #                 pca_feats[0][3],pca_feats[0][4],
                           rgb_sigma_est[0],rgb_sigma_est[1],rgb_sigma_est[2]
        )


# In[11]:


cols = ['pca0','pca1', 'pca2',
#        'pca3','pca4',
        's1','s2','s3']

for col in cols:
    train[col] = None
    test[col] = None


# In[12]:


NCORE = 8

color_info = mp.Manager().dict()

# Using a queue since the image read is a bottleneck
q = mp.Queue(maxsize=NCORE)
iolock = mp.Lock()
pool = mp.Pool(NCORE, initializer=color_stats, initargs=(q, iolock))

for i in train_images:
    q.put(i)  # blocks until q below its max size

for i in test_images:
    q.put(i)  # blocks until q below its max size
    
# tell workers we're done
for _ in range(NCORE):  
    q.put(None)
pool.close()
pool.join()

color_info = dict(color_info)


# In[13]:


for n, col in enumerate(cols):
    train[col] = train['fname'].apply(lambda x: color_info[x][n])
    test[col] = test['fname'].apply(lambda x: color_info[x][n])


# In[14]:


y = train['camera'].values
X_train = train[cols].values
X_test = test[cols].values
clf = RandomForestClassifier(n_estimators=5)
# clf = SVC(decision_function_shape='ovo',kernel='rbf')
clf.fit(X_train, y)


# In[15]:





# In[15]:


y_pred = clf.predict(X_test)
subm = pd.read_csv(input_path / 'sample_submission.csv', index_col='fname')
subm['camera'] = y_pred
subm.to_csv('pca_svm_benchmark.csv')


# ### That's it! Thank you for reaching to the end and welcome to share your thought about PCA practice for this problem in comments!
