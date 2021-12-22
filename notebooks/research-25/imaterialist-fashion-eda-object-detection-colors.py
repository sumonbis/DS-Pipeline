
# coding: utf-8

# 
# ![](https://theknclan.com/wp-content/uploads/2017/10/635980679147435890-488367249_FashionHeader.png)
# 
# # Extensive EDA of iMaterialist (Fashion) Dataset with Object Detection and Color Analysis
# 
# This notebook contains the exploration of iMaterialist Challenge (Fashion) at FGVC5 [dataset](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018)
# 
# About the iMaterialist (Fashion) Competition - 
# 
# As shoppers move online, it would be a dream come true to have products in photos classified automatically. But, automatic product recognition is tough because for the same product, a picture can be taken in different lighting, angles, backgrounds, and levels of occlusion. Meanwhile different fine-grained categories may look very similar, for example, royal blue vs turquoise in color. Many of today’s general-purpose recognition machines simply cannot perceive such subtle differences between photos, yet these differences could be important for shopping decisions.
# 
# Tackling issues like this is why the Conference on Computer Vision and Pattern Recognition (CVPR) has put together a workshop specifically for data scientists focused on fine-grained visual categorization called the FGVC5 workshop. As part of this workshop, CVPR is partnering with Google, Wish, and Malong Technologies to challenge the data science community to help push the state of the art in automatic image classification.
# 
# In this competition, FGVC workshop organizers with Wish and Malong Technologies challenge you to develop algorithms that will help with an important step towards automatic product detection – to accurately assign attribute labels for fashion images. Individuals/Teams with top submissions will be invited to present their work live at the FGVC5 workshop.  
# 
# 
# 
# 
# **Contents**
# 
# **1. Descriptive Statistics**   
# &nbsp;&nbsp;&nbsp;&nbsp;  1.1 Counts of Images and Labels  
# &nbsp;&nbsp;&nbsp;&nbsp;     1.2 Top Labels in the dataset  
# &nbsp;&nbsp;&nbsp;&nbsp;     1.3 Most Common Co-occuring Labels  
# &nbsp;&nbsp;&nbsp;&nbsp;     1.4 Images with maxium Labels  
# &nbsp;&nbsp;&nbsp;&nbsp;     1.5 Images with single Label  
# &nbsp;&nbsp;&nbsp;&nbsp;     1.6 Freq Dist of Images in different label count buckets  
# **2. Colors Used in the Images**     
# &nbsp;&nbsp;&nbsp;&nbsp;     2.1 Top Average Color of the images  
# &nbsp;&nbsp;&nbsp;&nbsp;     2.2 Dominant Colors present in the images  
# &nbsp;&nbsp;&nbsp;&nbsp;     2.3 Common Color Palletes    
# **3. Object Detection**  
# &nbsp;&nbsp;&nbsp;&nbsp;     3.1 Top Colors Detected in the images  
# &nbsp;&nbsp;&nbsp;&nbsp;     3.2 Top Objects Detected in the images  

# ## Dataset Preparation 

# In[ ]:


from IPython.core.display import HTML
from IPython.display import Image
from collections import Counter
import pandas as pd 
import json


from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from wordcloud import WordCloud
from plotly import tools
import seaborn as sns
from PIL import Image

import tensorflow as tf
import numpy as np

init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## read the dataset 

train_path = '../input/imaterialist-challenge-fashion-2018/train.json'
test_path = '../input/imaterialist-challenge-fashion-2018/test.json'
valid_path = '../input/imaterialist-challenge-fashion-2018/validation.json'

train_inp = open(train_path).read()
train_inp = json.loads(train_inp)

test_inp = open(test_path).read()
test_inp = json.loads(test_inp)

valid_inp = open(valid_path).read()
valid_inp = json.loads(valid_inp)


# ## 1. Descriptive Statistics
# 
# ## 1.1 How many Images and how many distinct labels are there in the dataset?

# In[ ]:


# how many images 
def get_stats(data):
    total_images = len(data['images'])

    all_annotations = []
    if 'annotations' in data:
        for each in data['annotations']:
            all_annotations.extend(each['labelId'])
    total_labels = len(set(all_annotations))
    return total_images, total_labels, all_annotations

total_images, total_labels, train_annotations = get_stats(train_inp)
print ("Total Images in the train:", total_images)
print ("Total Labels in the train:", total_labels)
print ("")

total_images, total_labels, test_annotations = get_stats(test_inp)
print ("Total Images in the test:", total_images)
print ("Total Labels in the test:", total_labels)
print ("")

total_images, total_labels, valid_annotations = get_stats(valid_inp)
print ("Total Images in the valid:", total_images)
print ("Total Labels in the valid:", total_labels)


# There are about 1 Million images provided in the train dataset and there are 228 distinct labels which are used to label these images. There are two other sources of data as well - test data and validation data but in thie notebook I have only used images from train dataset.
# 
# ## 1.2 Which are the top used Labels in the dataset ?

# In[ ]:


train_labels = Counter(train_annotations)

xvalues = list(train_labels.keys())
yvalues = list(train_labels.values())

trace1 = go.Bar(x=xvalues, y=yvalues, opacity=0.8, name="year count", marker=dict(color='rgba(20, 20, 20, 1)'))
layout = dict(width=800, title='Distribution of different labels in the train dataset', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# In[ ]:


valid_labels = Counter(valid_annotations)

xvalues = list(valid_labels.keys())
yvalues = list(valid_labels.values())

trace1 = go.Bar(x=xvalues, y=yvalues, opacity=0.8, name="year count", marker=dict(color='rgba(20, 20, 20, 1)'))
layout = dict(width=800, title='Distribution of different labels in the valid dataset', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# In[ ]:


def get_images_for_labels(labellist, data):
    image_ids = []
    for each in data['annotations']:
        if all(x in each['labelId'] for x in labellist):
            image_ids.append(each['imageId'])
            if len(image_ids) == 2:
                break
    image_urls = []
    for each in data['images']:
        if each['imageId'] in image_ids:
            image_urls.append(each['url'])
    return image_urls


# In[ ]:


# most common labels 

temps = train_labels.most_common(10)
labels_tr = ["Label-"+str(x[0]) for x in temps]
values = [x[1] for x in temps]

trace1 = go.Bar(x=labels_tr, y=values, opacity=0.7, name="year count", marker=dict(color='rgba(120, 120, 120, 0.8)'))
layout = dict(height=400, title='Top 10 Labels in the train dataset', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# Label 66 is the most used label with almost 750K images tagged with this label in the training dataset

# In[ ]:


temps = valid_labels.most_common(10)
labels_vl = ["Label-"+str(x[0]) for x in temps]
values = [x[1] for x in temps]

trace1 = go.Bar(x=labels_vl, y=values, opacity=0.7, name="year count", marker=dict(color='rgba(120, 120, 120, 0.8)'))
layout = dict(height=400, title='Top 10 Labels in the valid dataset', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# Again, in the validation dataset, Label 66 is the most used label but second most label used is label-17 not label-105 of training dataset

# ## 1.3 What are the most Common Co-Occuring Labels in the dataset
# 
# Since every image can be classified into multiple labels, it will be interesting to note which lables have co-occured together

# In[ ]:


# Most Commonly Occuring Labels 

def cartesian_reduct(alist):
    results = []
    for x in alist:
        for y in alist:
            if x == y:
                continue
            srtd = sorted([int(x),int(y)])
            srtd = " AND ".join([str(x) for x in srtd])
            results.append(srtd)
    return results 

co_occurance = []
for i, each in enumerate(train_inp['annotations']):
    prods = cartesian_reduct(each['labelId'])
    co_occurance.extend(prods)


# In[ ]:


coocur = Counter(co_occurance).most_common(10)
labels = list(reversed(["Label: "+str(x[0]) for x in coocur]))
values = list(reversed([x[1] for x in coocur]))

trace1 = go.Bar(x=values, y=labels, opacity=0.7, orientation="h", name="year count", marker=dict(color='rgba(130, 130, 230, 0.8)'))
layout = dict(height=400, title='Most Common Co-Occuring Labels in the dataset', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# From the above graph, (label 66 and label 105) and (label 66 and label 171) have been used most number of times while labelling the images, with the total count of 460K and 445K respectively. Apart from the most frequently occuring label "66", label 105 and label 153 have been used repeatedly in the dataset.
# 
# ## 1.4 Which Images are tagged with Maximum Labels
# 
# Some images are labelled with single label but some images can have labels as high as 20. Lets get the images having the largest numbers of labels in the dataset

# In[ ]:


def get_image_url(imgid, data):
    for each in data['images']:
        if each['imageId'] == imgid:
            return each['url']

srtedlist = sorted(train_inp['annotations'], key=lambda d: len(d['labelId']), reverse=True)


# In[ ]:


for img in srtedlist[:5]:
    iurl = get_image_url(img['imageId'], train_inp)  
    labelpair = ", ".join(img['labelId'])
    imghtml = """Labels: """+ str(labelpair) +""" &nbsp;&nbsp; <b>Total Labels: """+ str(len(img['labelId'])) + """</b><br>""" + "<img src="+iurl+" width=200px; style='float:left'>"
    display(HTML(imghtml))


# ## 1.5 Which Images have perfect label ie. a Single Label
# 
# Lets get some of the images which has only one label

# In[ ]:


# How many images are labelled with only 1 label 
for img in srtedlist[-5:]:
    iurl = get_image_url(img['imageId'], train_inp)  
    labelpair = ", ".join(img['labelId'])
    imghtml = """<b> Label: """+ str(labelpair) +"""</b><br>""" + "<img src="+iurl+" width=200px; height=200px; style='float:left'>"
    display(HTML(imghtml))


# ## 1.6 Frequency Distribution of Images with respective Labels Counts in the dataset
# 
# Lets visualize how many images are there in each label count bucket. 

# In[ ]:


lbldst = Counter([len(x['labelId']) for x in srtedlist])

labels = list(lbldst.keys())
values = list(lbldst.values())

trace1 = go.Bar(x=labels, y=values, opacity=0.7, name="year count", marker=dict(color='rgba(10, 80, 190, 0.8)'))
layout = dict(height=400, title='Frequency distribution of images with respective labels counts ', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# Most of the images in the dataset have 5 or 6 labels on an average. 
# 
# ## 2. Colors Used in the Images 
# 
# In the e-commerce industry, colors play a very important role in the customer behaviours. Some people are more inclined towards soft colors while some prefer warm colors. In this section, lets visualize what type of colors are used in the images. 
# 
# ## 2.1 Common Average Color of the Images 

# In[ ]:


import urllib
from io import StringIO

def compute_average_image_color(img):
    width, height = img.size
    count, r_total, g_total, b_total = 0, 0, 0, 0
    for x in range(0, width):
        for y in range(0, height):
            r, g, b = img.getpixel((x,y))
            r_total += r
            g_total += g
            b_total += b
            count += 1
    return (r_total/count, g_total/count, b_total/count)


# In[ ]:


import os 
imgpath = '../input/sampleimages/top_images/top_images/'
read_from_disk = True

if read_from_disk:
    srtedlist = os.listdir(imgpath)
else:
    srtedlist = sorted(inp['annotations'], key=lambda d: len(d['labelId']), reverse=True)


# In[ ]:


average_colors = {}
for img in srtedlist[:10]:
    if read_from_disk:
        img = Image.open(imgpath + img)
    else:
        iurli = get_image_url(img['imageId'])

        ## download the images 
        # filename = iurli.split("/")[-1].split("-large")[0]
        # urllib.urlretrieve(iurli, "top_images/"+filename)
        
        file = cStringIO.StringIO(urllib.urlopen(iurli).read())
        img = Image.open(img)
           
    average_color = compute_average_image_color(img)
    if average_color not in average_colors:
        average_colors[average_color] = 0
    average_colors[average_color] += 1


# In[ ]:


for average_color in average_colors:
    average_color1 = (int(average_color[0]),int(average_color[1]),int(average_color[2]))
    image_url = "<span style='display:inline-block; min-width:200px; background-color:rgb"+str(average_color1)+";padding:10px 10px;'>"+str(average_color1)+"</span>"
#     print (image_url)
    display(HTML(image_url))


# ## 2.2 Most Dominant Colors Used in the Images 

# In[ ]:


## top used colors in images 
from colorthief import ColorThief
import urllib 

pallets = []
for img in srtedlist[:10]:
    
    if read_from_disk:
        img = imgpath + img
    else:
        iurli = get_image_url(img['imageId'])

        ## download the images 
        # filename = iurli.split("/")[-1].split("-large")[0]
        # urllib.urlretrieve(iurli, "top_images/"+filename)
        
        file = cStringIO.StringIO(urllib.urlopen(iurli).read())
        img = Image.open(img)

    color_thief = ColorThief(img)
    dominant_color = color_thief.get_color(quality=1)
    
    image_url = "<span style='display:inline-block; min-width:200px; background-color:rgb"+str(dominant_color)+";padding:10px 10px;'>"+str(dominant_color)+"</span>"
    display(HTML(image_url))
    
    palette = color_thief.get_palette(color_count=6)
    pallets.append(palette)


# ## 2.3 Common Color Pallets of the Images

# In[ ]:


for pallet in pallets:
    img_url = ""
    for pall in pallet:
        img_url += "<span style='background-color:rgb"+str(pall)+";padding:20px 10px;'>"+str(pall)+"</span>"
    img_url += "<br>"
    display(HTML(img_url))
    print 
    


# ## 3. Object Detection using TensorFlow API 
# 
# 
# I have used tensorflow API for object detection the code is given in the following cell.
# 
# 
# ![title](https://github.com/tensorflow/models/raw/master/research/object_detection/g3doc/img/kites_detections_output.jpg)

# In[ ]:


### UNCOMMENT THE FOLLOWING LINE AFTER DOWNLOADING THE UTILS FROM THIS LINK - https://github.com/tensorflow/models/tree/master/research/object_detection/utils

# from utils import label_map_util

def DOWNLOAD_MODELS():
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

def detect_object(filename):

    def img2array(img):
        (img_width, img_height) = img.size
        return np.array(img.getdata()).reshape((img_width, img_height, 3)).astype(np.uint8)

    categories, probabilities = [], []
    PATH_TO_CKPT = 'frozen_inference_graph.pb'
    PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image = Image.open(filename)
            image_np = img2array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            for index,value in enumerate(classes[0]):
                if float(scores[0,index]) > 0.1:
                    temp =  category_index.get(value)['name']
                    if temp not in categories:
                        categories.append(temp)
                        probabilities.append(scores[0,index])
    return categories, probabilities


# In[ ]:


## UNCOMMENT THE FOLLOWING LINES TO RUN THE OBJECT DETECTION MODEL AND SAVE THE RESULTS 

# for img in srtedlist[:10]:
#     iurli = get_image_url(img['imageId'])
    
#     file = cStringIO.StringIO(urllib.urlopen(iurli).read())
#     objects = detect_object(file)


# - Reference: [TensorFlow Object Detection Notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)  
# - Pre-Trained Models Reference: [PreTrained Models](https://github.com/tensorflow/models/tree/676a4f70c20020ed41b533e0c331f115eeffe9a3/research/object_detection)  
# - Link to download the Utils: https://github.com/tensorflow/models/tree/master/research/object_detection/utils

# Since it would have taken a lot of time on kaggle kernals, I have pre-computed the objects in my local machine.

# In[ ]:


objpath = '../input/precomputedobjects/objects.txt'

objs = open(objpath).read().strip().split("\n")
colors = [_ for _ in objs if "color" in _]
non_colors = [_ for _ in objs if "color" not in _]


# ## 3.1 Top Objects detected using Object detection 

# In[ ]:


txt = ""
for i, color in enumerate(Counter(non_colors).most_common(100)):
    txt += color[0]+" "
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(txt)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top Objects Detected in the images", fontsize=15)
plt.axis("off")
plt.show() 


# ## 3.2 Top Color Detected in the images

# In[ ]:


txt = ""
for i, color in enumerate(Counter(colors).most_common(100)):
    txt += (color[0] + " ")
txt = txt.replace("color", " ")
wordcloud = WordCloud(max_font_size=50, width=600, height=300, background_color='white').generate(txt)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top Colors Used in the images", fontsize=15)
plt.axis("off")
plt.show() 


# Thanks for viewing the notebook. Hope You liked it, if liked it please upvote. 
