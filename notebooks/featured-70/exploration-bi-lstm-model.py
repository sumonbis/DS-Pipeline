#!/usr/bin/env python
# coding: utf-8

# ![](http://)<H1>Problem Understanding</H1>
# 
# The YouTube8M challenge is a multi-class classification problem, where we are asked to predict for each video, given video & frame level audio and frame RGB features, to which group of categories it belongs to.
# 
# I have divided entire task into two parts
# 
# 1. Simple Data Exploration,  Labels/classes study of sample videos. 
# 2. Created a Bi-LSTM multilabel neural model by randomly created sample data.
# 
# Lets first explore the labels for the training data, their distribution and frequent patterns and co-occurance of the most frequent label categories.
# 
# **Since we have been given sample dataset  here, so all my exploration will be done on sample data, we can do the same anlaysis on large corpus using GCloud ML Engine**

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import plotly.plotly as py

import os
print(os.listdir("../input"))
# video level feature file
print(os.listdir("../input/video"))
# frame level features file
print(os.listdir("../input/frame"))


# <H3>File descriptions</H3>
# 1. ** labels_names_2018.csv**:  a mapping between label_id and label_name <br>
# 
# 1. **vocabularu.csv:** :  the full data dictionary for label names and their descriptions <br>
# 
# 1. ** video (video-level data):**  contains video level info for each video, these files are in TFRecords format, will explore it later let us first explore what features it has:<br>
#     a. `id`: unique id for the video, in train set it is a Youtube video id, and in test/validation they are anonymized<br>
#     b. `labels`: list of labels of that video<br>
#     c. `mean_rgb`: float array of length 1024<br>
#     d. `mean_audio`: float array of length 128<br>
# 
# 1. **frame (frame-level data) :** contains frame level info for each video, again files are given in TFRecords format, lets see features <br>
#     a. `id`: unique id for the video, in train set it is a YouTube video id, and in test/validation they are anonymized.<br>
#     b. `labels`: list of labels of that video. <br>
#     c. `rgb`: Each frame has float array of length 1024,<br>
#     d. `audio`: Each frame has float array of length 128<br>
#  
# 1. **sample_submission.csv ** a sample submission file in the correct format, each row has <br>
#    a. `VideoId` - the id of the video<br>
#    b. `LabelConfidencePairs`: space delimited label/prediction pairs ordered by descending confidence
# 

# <H2>EDA </H2>

# <h3>Let us first explore labels and their distributions</h3>

# In[3]:


# total number of labels
labels_df = pd.read_csv('../input/label_names_2018.csv')
print(labels_df.head())
print("Total nubers of labels in sample dataset: %s" %(len(labels_df['label_name'].unique())))


# <h4>Exploring video level data</h4>

# In[4]:


# distribution of labels
video_files = ["../input/video/{}".format(i) for i in os.listdir("../input/video")]
print(video_files)

vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for file in video_files:
    for example in tf.python_io.tf_record_iterator(file):
        tf_example = tf.train.Example.FromString(example)

        vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
        labels.append(tf_example.features.feature['labels'].int64_list.value)
        mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
        mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)

print('Number of videos in Sample data set: %s' % str(len(vid_ids)))
print('Picking a youtube video id: %s' % vid_ids[13])
print('List of label ids for youtube video id %s, are - %s' % (vid_ids[13], str(labels[13])))
print('First 20 rgb feature of a youtube video (',vid_ids[13],'): are - %s' % str(mean_rgb[13][:20]))


# > <h2>Lets have a look at the most common labels and the relation among thems </h2>

# In[5]:


# Lets convert labels for each video into their respective names
labels_name = []
for row in labels:
    n_labels = []
    for label_id in row:
        # some labels ids are missing so have put try/except
        try:
            n_labels.append(str(labels_df[labels_df['label_id']==label_id]['label_name'].values[0]))
        except:
            continue
    labels_name.append(n_labels)

print('List of label names for youtube video id %s, are - %s' % (vid_ids[13], str(labels_name[13])))


# In[6]:


# creating labels count dictionary
from collections import Counter
import operator

all_labels = []
for each in labels_name:
    all_labels.extend(each)

labels_count_dict = dict(Counter(all_labels))


# Lets have a look at distribution of top 25 labels

# In[7]:


# creating label count dataframe
labels_count_df = pd.DataFrame.from_dict(labels_count_dict, orient='index').reset_index()
labels_count_df.columns = ['label', 'count']
sorted_labels_count_df = labels_count_df.sort_values('count', ascending=False)

# plotting top 25 labels distribution
TOP = 25
TOP_labels = list(sorted_labels_count_df['label'])[:TOP]
fig, ax = plt.subplots(figsize=(10,7))
sns.barplot(y='label', x='count', data=sorted_labels_count_df.iloc[0:TOP, :])
plt.title('Top {} labels with sample count'.format(TOP))


# Lets explore most common occuring labels with these top 25 labels.

# In[8]:


# creating common occurs labels count dict
common_occur_top_label_dict = {}
for row in labels_name:
    for label in row:
        if label in TOP_labels:
            c_labels = [label + "|" + x for x in row if x != label]
            for c_label in c_labels: 
                common_occur_top_label_dict[c_label] = common_occur_top_label_dict.get(c_label, 0) + 1

# creating dataframe
common_occur_top_label_df = pd.DataFrame.from_dict(common_occur_top_label_dict, orient='index').reset_index()
common_occur_top_label_df.columns = ['common_label', 'count']
sorted_common_occur_top_label_df = common_occur_top_label_df.sort_values('count', ascending=False)


# plotting 25 common occurs labels from top labels
TOP = 25
fig, ax = plt.subplots(figsize=(10,7))
sns.barplot(y='common_label', x='count', data=sorted_common_occur_top_label_df.iloc[0:TOP, :])
plt.title('Top {} common occur labels with sample count'.format(TOP))


# This shows game and vehicle are most commonly occurs labels among sample youtube videos

# <H2> Create Network Graph For Top Labels</H2>

# In[9]:


# libraries
import pandas as pd
import numpy as np
 
top_cooccurance_label_dict = {}
for row in labels_name:
    for label in row:
        if label in TOP_labels:
            top_label_siblings = [x for x in row if x != label]
            for sibling in top_label_siblings:
                if label not in top_cooccurance_label_dict:
                    top_cooccurance_label_dict[label] = {}
                top_cooccurance_label_dict[label][sibling] = top_cooccurance_label_dict.get(label, {}).get(sibling, 0) + 1

from_label= []
to_label = []
value = []
for key, val in top_cooccurance_label_dict.items():
    for key2, val2 in val.items():
        from_label.append(key)
        to_label.append(key2)
        value.append(val2)

df = pd.DataFrame({ 'from': from_label, 'to': to_label, 'value': value})
sorted_df = df.sort_values('value', ascending=False)
sorted_df = sorted_df.iloc[:50, ]


# In[ ]:


node_colors = ['turquoise', 'turquoise', 'green', 'crimson', 'grey', 'turquoise', 'turquoise', 
'grey', 'skyblue', 'crimson', 'yellow', 'green', 'turquoise', 
'skyblue', 'skyblue', 'green', 'green', 'lightcoral', 'grey', 'yellow', 
'turquoise', 'skyblue', 'orange', 'green', 'skyblue', 'green', 'turquoise', 
'lightcoral', 'yellow', 'lightcoral', 'green', 'turquoise', 'lightcoral', 'turquoise', 
'yellow', 'orange', 'lightcoral', 'grey', 'green', 'orange', 'crimson', 
'skyblue', 'lightcoral', 'lightcoral', 'skyblue', 'crimson', 'yellow', 'yellow', 'lightcoral', 
'yellow']


# In[19]:


import networkx as nx
import matplotlib.pyplot as plt
 
df = sorted_df
# Build your graph
G=nx.from_pandas_dataframe(df, 'from', 'to', 'value', create_using=nx.Graph() )
plt.figure(figsize = (10,10))
nx.draw(G, pos=nx.circular_layout(G), node_size=1000, with_labels=True, node_color=node_colors)
nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels=nx.get_edge_attributes(G, 'value'))
plt.title('Network graph representing the co-occurance between the categories', size=20)
plt.show()


# 
# <H3> let's  Explore frame-level data for videos </H3>

# In[20]:


frame_files = ["../input/frame/{}".format(i) for i in os.listdir("../input/frame")]
feat_rgb = []
feat_audio = []

for file in frame_files:
    for example in tf.python_io.tf_record_iterator(file):        
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
        sess = tf.InteractiveSession()
        rgb_frame = []
        audio_frame = []
        # iterate through frames
        for i in range(n_frames):
            rgb_frame.append(tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                           ,tf.float32).eval())
            audio_frame.append(tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                           ,tf.float32).eval())


        sess.close()
        feat_rgb.append(rgb_frame)
        feat_audio.append(audio_frame)
        break


# In[217]:


print("No. of videos %d" % len(feat_rgb))
print('The first video has %d frames' %len(feat_rgb[0]))
print("Max frame length is: %d" % max([len(x) for x in feat_rgb]))


# <H1> Bi-LSTM Multilabel classification </H1>

# **Here we will be using deep learning model with below architecture, since frames are sequence data, we will be utilising bi-directional lstm to learn this frame data and merge their ourput with video level data which later will pass through output sigmoid layer with units equal to no. of features**

# **Link Diagram ** - https://drive.google.com/file/d/1mGnPBya9eyKj0ZP6a4GUuFVXSr7pRw4j/view?usp=sharing

# In[15]:


# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
import operator
import time 
import gc
import os


# **creating training and dev set**

# In[16]:


def create_train_dev_dataset(video_rgb, video_audio, frame_rgb, frame_audio, labels):
    """
    Method to created training and validation data
    """
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    video_rgb_shuffled = video_rgb[shuffle_indices]
    video_audio_shuffled = video_audio[shuffle_indices]
    frame_rgb_shuffled = frame_rgb[shuffle_indices]
    frame_audio_shuffled = frame_audio[shuffle_indices]
    labels_shuffled = labels[shuffle_indices]

    dev_idx = max(1, int(len(labels_shuffled) * validation_split_ratio))

    del video_rgb
    del video_audio
    del frame_rgb
    del frame_audio
    gc.collect()

    train_video_rgb, val_video_rgb = video_rgb_shuffled[:-dev_idx], video_rgb_shuffled[-dev_idx:]
    train_video_audio, val_video_audio = video_audio_shuffled[:-dev_idx], video_audio_shuffled[-dev_idx:]
    
    train_frame_rgb, val_frame_rgb = frame_rgb_shuffled[:-dev_idx], frame_rgb_shuffled[-dev_idx:]
    train_frame_audio, val_frame_audio = frame_audio_shuffled[:-dev_idx], frame_audio_shuffled[-dev_idx:]
    
    train_labels, val_labels = labels_shuffled[:-dev_idx], labels_shuffled[-dev_idx:]
    
    del video_rgb_shuffled, video_audio_shuffled, frame_rgb_shuffled, frame_audio_shuffled, labels_shuffled
    gc.collect()
    
    return (train_video_rgb, train_video_audio, train_frame_rgb, train_frame_audio, train_labels, val_video_rgb, val_video_audio, 
            val_frame_rgb, val_frame_audio, val_labels)
    


# **Defining Model parameters and creating architecture**

# In[17]:


max_frame_rgb_sequence_length = 10
frame_rgb_embedding_size = 1024

max_frame_audio_sequence_length = 10
frame_audio_embedding_size = 128

number_dense_units = 1000
number_lstm_units = 100
rate_drop_lstm = 0.2
rate_drop_dense = 0.2
activation_function='relu'
validation_split_ratio = 0.2
label_feature_size = 10

def create_model(video_rgb, video_audio, frame_rgb, frame_audio, labels):
    """Create and store best model at `checkpoint` path ustilising bi-lstm layer for frame level data of videos"""
    train_video_rgb, train_video_audio, train_frame_rgb, train_frame_audio, train_labels, val_video_rgb, val_video_audio, val_frame_rgb, val_frame_audio, val_labels = create_train_dev_dataset(video_rgb, video_audio, frame_rgb, frame_audio, labels) 
    
    # Creating 2 bi-lstm layer, one for rgb and other for audio level data
    lstm_layer_1 = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
    lstm_layer_2 = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
    
    # creating input layer for frame-level data
    frame_rgb_sequence_input = Input(shape=(max_frame_rgb_sequence_length, frame_rgb_embedding_size), dtype='float32')
    frame_audio_sequence_input = Input(shape=(max_frame_audio_sequence_length, frame_audio_embedding_size), dtype='float32')
    
    frame_x1 = lstm_layer_1(frame_rgb_sequence_input)
    frame_x2 = lstm_layer_2(frame_audio_sequence_input)
    
    # creating input layer for video-level data
    video_rgb_input = Input(shape=(video_rgb.shape[1],))
    video_rgb_dense = Dense(int(number_dense_units/2), activation=activation_function)(video_rgb_input)
    
    video_audio_input = Input(shape=(video_audio.shape[1],))
    video_audio_dense = Dense(int(number_dense_units/2), activation=activation_function)(video_audio_input)
    
    # merging frame-level bi-lstm output and later passed to dense layer by applying batch-normalisation and dropout
    merged_frame = concatenate([frame_x1, frame_x2])
    merged_frame = BatchNormalization()(merged_frame)
    merged_frame = Dropout(rate_drop_dense)(merged_frame)
    merged_frame_dense = Dense(int(number_dense_units/2), activation=activation_function)(merged_frame)
    
    # merging video-level dense layer output
    merged_video = concatenate([video_rgb_dense, video_audio_dense])
    merged_video = BatchNormalization()(merged_video)
    merged_video = Dropout(rate_drop_dense)(merged_video)
    merged_video_dense = Dense(int(number_dense_units/2), activation=activation_function)(merged_video)
    
    # merging frame-level and video-level dense layer output
    merged = concatenate([merged_frame_dense, merged_video_dense])
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
     
    merged = Dense(number_dense_units, activation=activation_function)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
    preds = Dense(label_feature_size, activation='sigmoid')(merged)
    
    model = Model(inputs=[frame_rgb_sequence_input, frame_audio_sequence_input, video_rgb_input, video_audio_input], outputs=preds)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    STAMP = 'lstm_%d_%d_%.2f_%.2f' % (number_lstm_units, number_dense_units, rate_drop_lstm, rate_drop_dense)

    checkpoint_dir = 'checkpoints/' + str(int(time.time())) + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
    tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))
    
    model.fit([train_frame_rgb, train_frame_audio, train_video_rgb, train_video_audio], train_labels,
              validation_data=([val_frame_rgb, val_frame_audio, val_video_rgb, val_video_audio], val_labels),
              epochs=200, batch_size=64, shuffle=True, callbacks=[early_stopping, model_checkpoint, tensorboard])    
    return model


# **Creating random data set for training **
# 
# ![](http://)Here I am creating a sample dataset of same size and dimension of training sample and will train the model

# In[18]:


import numpy as np
import random

sample_length = 1000

video_rgb = np.random.rand(sample_length, 1024)
video_audio = np.random.rand(sample_length, 128)

frame_rgb = np.random.rand(sample_length, 10, 1024)
frame_audio = np.random.rand(sample_length, 10, 128)

# Here I have considered i have only 10 labels
labels = np.zeros([sample_length,10])
for i in range(len(labels)):
    j = random.randint(0,9)
    labels[i][j] = 1 


# <H2> Training Model </H2>

# In[6]:


model = create_model(video_rgb, video_audio, frame_rgb, frame_audio, labels)


# <H3>Testing with created random test data</H3>

# In[19]:


test_video_rgb = np.random.rand(1, 1024)
test_video_audio = np.random.rand(1, 128)

test_frame_rgb = np.random.rand(1, 10, 1024)
test_frame_audio = np.random.rand(1, 10, 128)

preds = list(model.predict([test_frame_rgb, test_frame_audio, test_video_rgb, test_video_audio], verbose=1).ravel())
index, value = max(enumerate(preds), key=operator.itemgetter(1))
print("Predicted Label - %s with probability - %s" % (str(index), str(value)))


# <H1> Thanks !! Hopes that help </H1>

# In[ ]:




