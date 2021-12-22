#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


get_ipython().system(' tar xf ../input/bird-songs-pad-and-resize-spectrogram/spectrograms_resized.tar.bz2')


# In[3]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import matplotlib.pyplot as plt

from fastai.vision import *


# In[4]:


data_dir = Path('../input')
label_dir = data_dir/'multilabel-bird-species-classification-nips2013/nips4b_bird_challenge_train_labels/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS'
spect_dir = Path('./spectrograms_resized')


# ## Clean up labels

# In[5]:


df = pd.read_csv(label_dir/'nips4b_birdchallenge_train_labels.csv', skiprows=[0, 2])
df.tail()


# In[6]:


df.drop(df.columns[[1, 3]],axis=1,inplace=True)  # drop colums 1 and 3
df.rename(columns={df.columns[0]:'file', df.columns[1]:'EMPTY'}, inplace=True)  # add 'EMPTY' label
df = df[:-1]  # drop last row
df.fillna(0, inplace=True)  # fill all the NaN's
df = df.astype('int32', errors='ignore') # make integer labels
df['file'] = df['file'].apply(lambda fn: str(Path(fn).with_suffix(''))) # drop .wav in the filenames

df.tail()


# ## Load dataset

# In[7]:


np.random.seed(42)
src = (ImageList.from_df(df, spect_dir, folder='train', suffix='.png')
                .split_by_rand_pct(0.2)
                .label_from_df(cols=df.columns[1:].tolist()))


# Don't do flips, it doens't make sense for spectrograms:

# In[8]:


tfms = get_transforms(do_flip=False, max_rotate=None, max_warp=None)
data = (src.transform(tfms, size=128)
        .databunch(num_workers=0).normalize(imagenet_stats))


# In[9]:


data.show_batch(rows=3, figsize=(12,9), ds_type=DatasetType.Valid)


# ## Train with size 128

# In[10]:


arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)
learn = cnn_learner(data, arch, metrics=acc_02, path='.')


# In[11]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[12]:


lr = 2.29E-02
learn.fit_one_cycle(5, slice(lr))


# In[13]:


learn.recorder.plot_losses()


# In[14]:


learn.save('stage-1-rn50', return_path=True)


# Unfreeze and fine tune:

# In[15]:


# learn.load('stage-1-rn50')


# In[16]:


learn.unfreeze()


# In[17]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[18]:


learn.fit_one_cycle(5, slice(3.02E-03, lr/5))


# In[19]:


learn.recorder.plot_losses()


# In[20]:


learn.save('stage-2-rn50')


# ## Train with size 256

# Replace data in learner with size 256 images

# In[21]:


data_256 = (src.transform(tfms, size=256)
            .databunch(num_workers=0).normalize(imagenet_stats))

learn.data = data_256
learn.data.train_ds[0][0].shape


# In[22]:


learn.freeze()


# In[23]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[24]:


lr = 5E-03
learn.fit_one_cycle(5, slice(lr))


# In[25]:


learn.recorder.plot_losses()


# In[26]:


learn.save('stage-1-256-rn50')


# Unfreeze and fine tune:

# In[27]:


learn.unfreeze()


# In[28]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[29]:


learn.fit_one_cycle(5, slice(1.58E-06, lr/5))


# In[30]:


learn.recorder.plot_losses()


# In[31]:


learn.save('stage-2-256-rn50')


# Run some more epochs

# In[32]:


learn.fit_one_cycle(5, slice(1.58E-06, lr/5))


# In[33]:


learn.recorder.plot_losses()


# In[34]:


learn.save('stage-2-256-rn50-10e')


# In[35]:


learn.export()


# ## Prediction and submission

# In[36]:


test = ImageList.from_folder(spect_dir/'test/')
len(test)


# In[37]:


predictor = load_learner('.', test=test, num_workers=0)
preds, _ = predictor.get_preds(ds_type=DatasetType.Test)
fpreds = preds[:, 1:].reshape(-1, )


# In[38]:


import itertools
names = [f.stem for f in predictor.data.test_ds.items]
fnames = [x + '.wav_classnumber_' + str(i) for x in names for i in range(1, len(data.classes))]


# In[39]:


test_df = pd.DataFrame({'ID':fnames, 'Probability':fpreds}, columns=['ID', 'Probability'])
test_df.to_csv('submission.csv', index=False)


# In[40]:


get_ipython().system(' rm -r $spect_dir')

