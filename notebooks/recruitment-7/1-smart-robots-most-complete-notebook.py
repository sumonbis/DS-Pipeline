
# coding: utf-8

# # CareerCon 2019 - Help Navigate Robots
# ## Robots are smart… by design !!
# 
# ![](https://www.lextronic.fr/imageslib/4D/0J7589.320.gif)
# 
# ---
# 
# Robots are smart… by design. To fully understand and properly navigate a task, however, they need input about their environment.
# 
# In this competition, you’ll help robots recognize the floor surface they’re standing on using data collected from Inertial Measurement Units (IMU sensors).
# 
# We’ve collected IMU sensor data while driving a small mobile robot over different floor surfaces on the university premises. The task is to predict which one of the nine floor types (carpet, tiles, concrete) the robot is on using sensor data such as acceleration and velocity. Succeed and you'll help improve the navigation of robots without assistance across many different surfaces, so they won’t fall down on the job.
# 
# ###  Its a golden chance to help humanity, by helping Robots !
# 
# <br>
# <img src="https://media2.giphy.com/media/EizPK3InQbrNK/giphy.gif" border="1" width="400" height="300">

# <br>
# # DATA

# **X_[train/test].csv** - the input data, covering 10 sensor channels and 128 measurements per time series plus three ID columns:
# 
# - ```row_id```: The ID for this row.
# 
# - ```series_id: ID``` number for the measurement series. Foreign key to y_train/sample_submission.
# 
# - ```measurement_number```: Measurement number within the series.
# 
# The orientation channels encode the current angles how the robot is oriented as a quaternion (see Wikipedia). Angular velocity describes the angle and speed of motion, and linear acceleration components describe how the speed is changing at different times. The 10 sensor channels are:
# 
# ```
# orientation_X
# 
# orientation_Y
# 
# orientation_Z
# 
# orientation_W
# 
# angular_velocity_X
# 
# angular_velocity_Y
# 
# angular_velocity_Z
# 
# linear_acceleration_X
# 
# linear_acceleration_Y
# 
# linear_acceleration_Z
# ```
# 
# **y_train.csv** - the surfaces for training set.
# 
# - ```series_id```: ID number for the measurement series.
# 
# - ```group_id```: ID number for all of the measurements taken in a recording session. Provided for the training set only, to enable more cross validation strategies.
# 
# - ```surface```: the target for this competition.
# 
# **sample_submission.csv** - a sample submission file in the correct format.

# ### Load packages

# In[1]:


import numpy as np 
import pandas as pd 
import os
from time import time
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
le = preprocessing.LabelEncoder()
from numba import jit
import itertools
from seaborn import countplot,lineplot, barplot
from numba import jit
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.style as style 
style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')
import gc
gc.enable()

get_ipython().system('ls ../input/')
get_ipython().system('ls ../input/robots-best-submission')
print ("Ready !")


# ### Load data

# In[2]:


data = pd.read_csv('../input/career-con-2019/X_train.csv')
tr = pd.read_csv('../input/career-con-2019/X_train.csv')
sub = pd.read_csv('../input/career-con-2019/sample_submission.csv')
test = pd.read_csv('../input/career-con-2019/X_test.csv')
target = pd.read_csv('../input/career-con-2019/y_train.csv')
print ("Data is ready !!")


# # Data exploration

# In[3]:


data.head()


# In[4]:


test.head()


# In[5]:


target.head()


# In[6]:


len(data.measurement_number.value_counts())


# Each series has 128 measurements. 
# 
# **1 serie = 128 measurements**. 
# 
# For example, serie with series_id=0 has a surface = *fin_concrete* and 128 measurements.

# ### describe (basic stats)

# In[7]:


data.describe()


# In[8]:


test.describe()


# In[9]:


target.describe()


# ### There is missing data in test and train data

# In[10]:


totalt = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])
print ("Missing Data at Training")
missing_data.tail()


# In[11]:


totalt = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])
print ("Missing Data at Test")
missing_data.tail()


# In[12]:


print ("Test has ", (test.shape[0]-data.shape[0])/128, "series more than Train (later I will prove it) = 768 registers")
dif = test.shape[0]-data.shape[0]
print ("Let's check this extra 6 series")
test.tail(768).describe()


# If we look at the features: orientation, angular velocity and linear acceleration, we can see big differences between **max** and **min** from entire test vs 6 extra test's series (see **linear_acceleration_Z**).
# 
# Obviously we are comparing 3810 series vs 6 series so this is not a big deal.

# ### goup_id will be important !!

# In[13]:


target.groupby('group_id').surface.nunique().max()


# In[14]:


target['group_id'].nunique()


# **73 groups**
# **Each group_id is a unique recording session and has only one surface type **

# In[15]:


sns.set(style='darkgrid')
sns.countplot(y = 'surface',
              data = target,
              order = target['surface'].value_counts().index)
plt.show()


# ### Target feature - surface and group_id distribution
# Let's show now the distribution of target feature - surface and group_id.
# by @gpreda.

# In[16]:


fig, ax = plt.subplots(1,1,figsize=(26,8))
tmp = pd.DataFrame(target.groupby(['group_id', 'surface'])['series_id'].count().reset_index())
m = tmp.pivot(index='surface', columns='group_id', values='series_id')
s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")
s.set_title('Number of surface category per group_id', size=16)
plt.show()


# We need to classify on which surface our robot is standing.
# 
# Multi-class Multi-output
# 
# 9 classes (suface)

# In[17]:


plt.figure(figsize=(23,5)) 
sns.set(style="darkgrid")
countplot(x="group_id", data=target, order = target['group_id'].value_counts().index)
plt.show()


# **So, we have 3810 train series, and 3816 test series.
# Let's engineer some features!**
# 
# ## Example: Series 1
# 
# Let's have a look at the values of features in a single time-series, for example series 1  ```series_id=0```
# 
# Click to see all measurements of the **first series** 

# In[18]:


serie1 = tr.head(128)
serie1.head()


# In[19]:


serie1.describe()


# In[20]:


plt.figure(figsize=(26, 16))
for i, col in enumerate(serie1.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.plot(serie1[col])
    plt.title(col)


# In this example, we can see a quite interesting performance:
# 1. Orientation X increases
# 2. Orientation Y decreases
# 3. We don't see any kind of pattern except for linear_acceleration_Y
# 
# And we know that in this series, the robot moved throuh "fine_concrete".

# In[21]:


target.head(1)


# In[22]:


del serie1
gc.collect()


# ## Visualizing Series
# 
# Before, I showed you as an example the series 1.
# 
# **This code allows you to visualize any series.**
# 
# From: *Code Snippet For Visualizing Series Id by @shaz13*

# In[23]:


series_dict = {}
for series in (data['series_id'].unique()):
    series_dict[series] = data[data['series_id'] == series]  


# In[24]:


def plotSeries(series_id):
    style.use('ggplot')
    plt.figure(figsize=(28, 16))
    print(target[target['series_id'] == series_id]['surface'].values[0].title())
    for i, col in enumerate(series_dict[series_id].columns[3:]):
        if col.startswith("o"):
            color = 'red'
        elif col.startswith("a"):
            color = 'green'
        else:
            color = 'blue'
        if i >= 7:
            i+=1
        plt.subplot(3, 4, i + 1)
        plt.plot(series_dict[series_id][col], color=color, linewidth=3)
        plt.title(col)


# **Now, Let's see code for series 15 ( is an example, try what you want)**

# In[25]:


id_series = 15
plotSeries(id_series)


# In[26]:


del series_dict
gc.collect()


# <br>
# ### Correlations (Part I)

# In[27]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(tr.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# **Correlations test (click "code")** 

# In[28]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(test.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# Well, this is immportant, there is a **strong correlation** between:
# - angular_velocity_Z and angular_velocity_Y
# - orientation_X and orientation_Y
# - orientation_Y and orientation_Z
# 
# Moreover, test has different correlations than training, for example:
# 
# - angular_velocity_Z and orientation_X: -0.1(training) and 0.1(test). Anyway, is too small in both cases, it should not be a problem.

# ## Fourier Analysis
# 
# My hope was, that different surface types yield (visible) differences in the frequency spectrum of the sensor measurements.
# 
# Machine learning techniques might learn frequency filters on their own, but why don't give the machine a little head start? So I computed the the cyclic FFT for the angular velocity and linear acceleration sensors and plotted mean and standard deviation of the absolute values of the frequency components per training surface category (leaving out the frequency 0 (i.e. constants like sensor bias, earth gravity, ...).
# 
# The sensors show some different frequency characterists (see plots below), but unfortunately the surface categories have all similar (to the human eye) shapes, varying mostly in total power, and the standard deviations are high (compared to differences in the means). So there are no nice strong characteristic peaks for surface types. But that does not mean, that there is nothing detectable by more sophisticated statistical methods.
# 
# This article http://www.kaggle.com/christoffer/establishing-sampling-frequency makes a convincing case, that the sampling frequency is around 400Hz, so according to that you would see the frequency range to 3-200 Hz in the diagrams (and aliased higher frequencies).
# 
# by [@trohwer64](https://www.kaggle.com/trohwer64)

# In[29]:


get_ipython().system('ls ../input')


# In[30]:


train_x = pd.read_csv('../input/career-con-2019/X_train.csv')
train_y = pd.read_csv('../input/career-con-2019/y_train.csv')


# In[31]:


import math

def prepare_data(t):
    def f(d):
        d=d.sort_values(by=['measurement_number'])
        return pd.DataFrame({
         'lx':[ d['linear_acceleration_X'].values ],
         'ly':[ d['linear_acceleration_Y'].values ],
         'lz':[ d['linear_acceleration_Z'].values ],
         'ax':[ d['angular_velocity_X'].values ],
         'ay':[ d['angular_velocity_Y'].values ],
         'az':[ d['angular_velocity_Z'].values ],
        })

    t= t.groupby('series_id').apply(f)

    def mfft(x):
        return [ x/math.sqrt(128.0) for x in np.absolute(np.fft.fft(x)) ][1:65]

    t['lx_f']=[ mfft(x) for x in t['lx'].values ]
    t['ly_f']=[ mfft(x) for x in t['ly'].values ]
    t['lz_f']=[ mfft(x) for x in t['lz'].values ]
    t['ax_f']=[ mfft(x) for x in t['ax'].values ]
    t['ay_f']=[ mfft(x) for x in t['ay'].values ]
    t['az_f']=[ mfft(x) for x in t['az'].values ]
    return t


# In[32]:


t=prepare_data(train_x)
t=pd.merge(t,train_y[['series_id','surface','group_id']],on='series_id')
t=t.rename(columns={"surface": "y"})


# In[33]:


def aggf(d, feature):
    va= np.array(d[feature].tolist())
    mean= sum(va)/va.shape[0]
    var= sum([ (va[i,:]-mean)**2 for i in range(va.shape[0]) ])/va.shape[0]
    dev= [ math.sqrt(x) for x in var ]
    return pd.DataFrame({
        'mean': [ mean ],
        'dev' : [ dev ],
    })

display={
'hard_tiles_large_space':'r-.',
'concrete':'g-.',
'tiled':'b-.',

'fine_concrete':'r-',
'wood':'g-',
'carpet':'b-',
'soft_pvc':'y-',

'hard_tiles':'r--',
'soft_tiles':'g--',
}


# In[34]:


import matplotlib.pyplot as plt
plt.figure(figsize=(14, 8*7))
#plt.margins(x=0.0, y=0.0)
#plt.tight_layout()
# plt.figure()

features=['lx_f','ly_f','lz_f','ax_f','ay_f','az_f']
count=0

for feature in features:
    stat= t.groupby('y').apply(aggf,feature)
    stat.index= stat.index.droplevel(-1)
    b=[*range(len(stat.at['carpet','mean']))]

    count+=1
    plt.subplot(len(features)+1,1,count)
    for i,(k,v) in enumerate(display.items()):
        plt.plot(b, stat.at[k,'mean'], v, label=k)
        # plt.errorbar(b, stat.at[k,'mean'], yerr=stat.at[k,'dev'], fmt=v)
   
    leg = plt.legend(loc='best', ncol=3, mode="expand", shadow=True, fancybox=True)
    plt.title("sensor: " + feature)
    plt.xlabel("frequency component")
    plt.ylabel("amplitude")

count+=1
plt.subplot(len(features)+1,1,count)
k='concrete'
v=display[k]
feature='lz_f'
stat= t.groupby('y').apply(aggf,feature)
stat.index= stat.index.droplevel(-1)
b=[*range(len(stat.at['carpet','mean']))]

plt.errorbar(b, stat.at[k,'mean'], yerr=stat.at[k,'dev'], fmt=v)
plt.title("sample for error bars (lz_f, surface concrete)")
plt.xlabel("frequency component")
plt.ylabel("amplitude")

plt.show()


# In[35]:


del train_x, train_y
gc.collect()


# ## Is it an Humanoid Robot instead of a car?
# 
# ![](https://media1.giphy.com/media/on7ipUR0rFjRS/giphy.gif)
# 
# **Acceleration**
# - X (mean at 0)
# - Y axis is centered at a value wich shows us the movement (straight ).
# - Z axis is centered at 10 (+- 9.8) wich is the gravity !! , you can see how the robot bounds.
# 
# Angular velocity (X,Y,Z) has mean (0,0,0) so there is no lineal movement on those axis (measured with an encoder or potentiometer)
# 
# **Fourier**
# 
# We can see: with a frequency 3 Hz we can see an acceleration, I think that acceleration represents one step.
# Maybe ee can suppose that every step is caused by many different movements, that's why there are different accelerations at different frequencies.
# 
# Angular velocity represents spins. 
# Every time the engine/servo spins, the robot does an step - relation between acc y vel.

# ---
# 
# # Feature Engineering

# In[36]:


def plot_feature_distribution(df1, df2, label1, label2, features,a=2,b=5):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(a,b,figsize=(17,9))

    for feature in features:
        i += 1
        plt.subplot(a,b,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();


# In[37]:


features = data.columns.values[3:]
plot_feature_distribution(data, test, 'train', 'test', features)


# Godd news, our basic features have the **same distribution (Normal) on test and training**. There are some differences between *orientation_X* , *orientation_Y* and *linear_acceleration_Y*.
# 
# I willl try **StandardScaler** to fix this, and remember: orientation , angular velocity and linear acceleration are measured with different units, scaling might be a good choice.

# In[38]:


def plot_feature_class_distribution(classes,tt, features,a=5,b=2):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(a,b,figsize=(16,24))

    for feature in features:
        i += 1
        plt.subplot(a,b,i)
        for clas in classes:
            ttc = tt[tt['surface']==clas]
            sns.kdeplot(ttc[feature], bw=0.5,label=clas)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();


# In[39]:


classes = (target['surface'].value_counts()).index
aux = data.merge(target, on='series_id', how='inner')
plot_feature_class_distribution(classes, aux, features)


# **Normal distribution**
# 
# There are obviously differences between *surfaces* and that's good, we will focus on that in order to classify them better.
# 
# Knowing this differences and that variables follow a normal distribution (in most of the cases) we need to add new features like: ```mean, std, median, range ...``` (for each variable).
# 
# However, I will try to fix *orientation_X* and *orientation_Y* as I explained before, scaling and normalizing data.
# 
# ---
# 
# ### Now with a new scale (more more precision)

# In[40]:


plt.figure(figsize=(26, 16))
for i,col in enumerate(aux.columns[3:13]):
    ax = plt.subplot(3,4,i+1)
    ax = plt.title(col)
    for surface in classes:
        surface_feature = aux[aux['surface'] == surface]
        sns.kdeplot(surface_feature[col], label = surface)


# ### Histogram for main features

# In[41]:


plt.figure(figsize=(26, 16))
for i, col in enumerate(data.columns[3:]):
    ax = plt.subplot(3, 4, i + 1)
    sns.distplot(data[col], bins=100, label='train')
    sns.distplot(test[col], bins=100, label='test')
    ax.legend()   


# ## Step 0 : quaternions

# Orientation - quaternion coordinates
# You could notice that there are 4 coordinates: X, Y, Z, W.
# 
# Usually we have X, Y, Z - Euler Angles. But Euler Angles are limited by a phenomenon called "gimbal lock," which prevents them from measuring orientation when the pitch angle approaches +/- 90 degrees. Quaternions provide an alternative measurement technique that does not suffer from gimbal lock. Quaternions are less intuitive than Euler Angles and the math can be a little more complicated.
# 
# Here are some articles about it:
# 
# http://www.chrobotics.com/library/understanding-quaternions
# 
# http://www.tobynorris.com/work/prog/csharp/quatview/help/orientations_and_quaternions.htm
# 
# Basically 3D coordinates are converted to 4D vectors.

# In[42]:


# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


# In[43]:


def fe_step0 (actual):
    
    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html
    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html
    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html
    
    # Spoiler: you don't need this ;)
    
    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)
    actual['mod_quat'] = (actual['norm_quat'])**0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']
    
    return actual


# 
# > *Are there any reasons to not automatically normalize a quaternion? And if there are, what quaternion operations do result in non-normalized quaternions?*
# 
# Any operation that produces a quaternion will need to be normalized because floating-point precession errors will cause it to not be unit length.
# I would advise against standard routines performing normalization automatically for performance reasons. 
# Any competent programmer should be aware of the precision issues and be able to normalize the quantities when necessary - and it is not always necessary to have a unit length quaternion.
# The same is true for vector operations.
# 
# source: https://stackoverflow.com/questions/11667783/quaternion-and-normalization

# In[44]:


data = fe_step0(data)
test = fe_step0(test)
print(data.shape)
data.head()


# In[45]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18, 5))

ax1.set_title('quaternion X')
sns.kdeplot(data['norm_X'], ax=ax1, label="train")
sns.kdeplot(test['norm_X'], ax=ax1, label="test")

ax2.set_title('quaternion Y')
sns.kdeplot(data['norm_Y'], ax=ax2, label="train")
sns.kdeplot(test['norm_Y'], ax=ax2, label="test")

ax3.set_title('quaternion Z')
sns.kdeplot(data['norm_Z'], ax=ax3, label="train")
sns.kdeplot(test['norm_Z'], ax=ax3, label="test")

ax4.set_title('quaternion W')
sns.kdeplot(data['norm_W'], ax=ax4, label="train")
sns.kdeplot(test['norm_W'], ax=ax4, label="test")

plt.show()


# ## Step 1: (x, y, z, w) -> (x,y,z)   quaternions to euler angles

# In[46]:


def fe_step1 (actual):
    """Quaternions to Euler Angles"""
    
    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual


# In[47]:


data = fe_step1(data)
test = fe_step1(test)
print (data.shape)
data.head()


# ![](https://d2gne97vdumgn3.cloudfront.net/api/file/UMYT4v0TyIgtyGm8ZXDQ)

# In[48]:


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))

ax1.set_title('Roll')
sns.kdeplot(data['euler_x'], ax=ax1, label="train")
sns.kdeplot(test['euler_x'], ax=ax1, label="test")

ax2.set_title('Pitch')
sns.kdeplot(data['euler_y'], ax=ax2, label="train")
sns.kdeplot(test['euler_y'], ax=ax2, label="test")

ax3.set_title('Yaw')
sns.kdeplot(data['euler_z'], ax=ax3, label="train")
sns.kdeplot(test['euler_z'], ax=ax3, label="test")

plt.show()


# **Euler angles** are really important, and we have a problem with Z.
# 
# ### Why Orientation_Z (euler angle Z) is so important?
# 
# We have a robot moving around, imagine a robot moving straight through different surfaces (each with different features), for example concrete and hard tile floor. Our robot can can **bounce** or **balance** itself a little bit on if the surface is not flat and smooth, that's why we need to work with quaternions and take care of orientation_Z.
# 
# ![](https://lifeboat.com/blog.images/robot-car-find-share-on-giphy.gif.gif)

# In[49]:


data.head()


# ## Step 2: + Basic features

# In[50]:


def feat_eng(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df
    


# In[51]:


get_ipython().run_cell_magic('time', '', 'data = feat_eng(data)\ntest = feat_eng(test)\nprint ("New features: ",data.shape)')


# In[52]:


data.head()


# ## New advanced features

# **Useful functions**

# In[53]:


from scipy.stats import kurtosis
from scipy.stats import skew

def _kurtosis(x):
    return kurtosis(x)

def CPT5(x):
    den = len(x)*np.exp(np.std(x))
    return sum(np.exp(x))/den

def skewness(x):
    return skew(x)

def SSC(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    xn_i1 = x[0:len(x)-2]  # xn-1
    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)
    return sum(ans[1:]) 

def wave_length(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    return sum(abs(xn_i2-xn))
    
def norm_entropy(x):
    tresh = 3
    return sum(np.power(abs(x),tresh))

def SRAV(x):    
    SRA = sum(np.sqrt(abs(x)))
    return np.power(SRA/len(x),2)

def mean_abs(x):
    return sum(abs(x))/len(x)

def zero_crossing(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1
    return sum(np.heaviside(-xn*xn_i2,0))


# This advanced features based on robust statistics.

# In[54]:


def fe_advanced_stats(data):
    
    df = pd.DataFrame()
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        if 'orientation' in col:
            continue
            
        print ("FE on column ", col, "...")
        
        df[col + '_skew'] = data.groupby(['series_id'])[col].skew()
        df[col + '_mad'] = data.groupby(['series_id'])[col].mad()
        df[col + '_q25'] = data.groupby(['series_id'])[col].quantile(0.25)
        df[col + '_q75'] = data.groupby(['series_id'])[col].quantile(0.75)
        df[col + '_q95'] = data.groupby(['series_id'])[col].quantile(0.95)
        df[col + '_iqr'] = df[col + '_q75'] - df[col + '_q25']
        df[col + '_CPT5'] = data.groupby(['series_id'])[col].apply(CPT5) 
        df[col + '_SSC'] = data.groupby(['series_id'])[col].apply(SSC) 
        df[col + '_skewness'] = data.groupby(['series_id'])[col].apply(skewness)
        df[col + '_wave_lenght'] = data.groupby(['series_id'])[col].apply(wave_length)
        df[col + '_norm_entropy'] = data.groupby(['series_id'])[col].apply(norm_entropy)
        df[col + '_SRAV'] = data.groupby(['series_id'])[col].apply(SRAV)
        df[col + '_kurtosis'] = data.groupby(['series_id'])[col].apply(_kurtosis) 
        df[col + '_zero_crossing'] = data.groupby(['series_id'])[col].apply(zero_crossing) 
        
    return df


# - Frequency of the max value
# - Frequency of the min value
# - Count Positive values
# - Count Negative values
# - Count zeros

# In[55]:


basic_fe = ['linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z',
           'angular_velocity_X','angular_velocity_Y','angular_velocity_Z']


# In[56]:


def fe_plus (data):
    
    aux = pd.DataFrame()
    
    for serie in data.index:
        #if serie%500 == 0: print ("> Serie = ",serie)
        
        aux = X_train[X_train['series_id']==serie]
        
        for col in basic_fe:
            data.loc[serie,col + '_unq'] = aux[col].round(3).nunique()
            data.loc[serie,col + 'ratio_unq'] = aux[col].round(3).nunique()/18
            try:
                data.loc[serie,col + '_freq'] = aux[col].value_counts().idxmax()
            except:
                data.loc[serie,col + '_freq'] = 0
            
            data.loc[serie,col + '_max_freq'] = aux[aux[col] == aux[col].max()].shape[0]
            data.loc[serie,col + '_min_freq'] = aux[aux[col] == aux[col].min()].shape[0]
            data.loc[serie,col + '_pos_freq'] = aux[aux[col] >= 0].shape[0]
            data.loc[serie,col + '_neg_freq'] = aux[aux[col] < 0].shape[0]
            data.loc[serie,col + '_nzeros'] = (aux[col]==0).sum(axis=0)


# ### Important !
# As you can see in this kernel https://www.kaggle.com/anjum48/leakage-within-the-train-dataset
# 
# As discussed in the discussion forums (https://www.kaggle.com/c/career-con-2019/discussion/87239#latest-508136) it looks as if each series is part of longer aquisition periods that have been cut up into chunks with 128 samples.
# 
# This means that each series is not truely independent and there is leakage between them via the orientation data. Therefore if you have any features that use orientation, you will get a very high CV score due to this leakage in the train set.
# 
# [This kernel](https://www.kaggle.com/anjum48/leakage-within-the-train-dataset) will show you how it is possible to get a CV score of 0.992 using only the **orientation data**.
# 
# ---
# 
# **So I recommend not to use orientation information**

# ## Correlations (Part II)

# In[57]:


#https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
corr_matrix = data.corr().abs()
raw_corr = data.corr()

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
top_corr = pd.DataFrame(sol).reset_index()
top_corr.columns = ["var1", "var2", "abs corr"]
# with .abs() we lost the sign, and it's very important.
for x in range(len(top_corr)):
    var1 = top_corr.iloc[x]["var1"]
    var2 = top_corr.iloc[x]["var2"]
    corr = raw_corr[var1][var2]
    top_corr.at[x, "raw corr"] = corr


# In[58]:


top_corr.head(15)


# ### Filling missing NAs and infinite data ∞  by zeroes 0

# In[59]:


data.fillna(0,inplace=True)
test.fillna(0,inplace=True)
data.replace(-np.inf,0,inplace=True)
data.replace(np.inf,0,inplace=True)
test.replace(-np.inf,0,inplace=True)
test.replace(np.inf,0,inplace=True)


# ## Label encoding

# In[60]:


target.head()


# In[61]:


target['surface'] = le.fit_transform(target['surface'])


# In[62]:


target['surface'].value_counts()


# In[63]:


target.head()


# # Run Model

# **use random_state at Random Forest**
# 
# if you don't use random_state you will get a different solution everytime, sometimes you will be lucky, but other times you will lose your time comparing.

# **Validation Strategy: Stratified KFold**

# In[64]:


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)


# In[65]:


predicted = np.zeros((test.shape[0],9))
measured= np.zeros((data.shape[0]))
score = 0


# In[66]:


for times, (trn_idx, val_idx) in enumerate(folds.split(data.values,target['surface'].values)):
    model = RandomForestClassifier(n_estimators=500, n_jobs = -1)
    #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)
    model.fit(data.iloc[trn_idx],target['surface'][trn_idx])
    measured[val_idx] = model.predict(data.iloc[val_idx])
    predicted += model.predict_proba(test)/folds.n_splits
    score += model.score(data.iloc[val_idx],target['surface'][val_idx])
    print("Fold: {} score: {}".format(times,model.score(data.iloc[val_idx],target['surface'][val_idx])))

    importances = model.feature_importances_
    indices = np.argsort(importances)
    features = data.columns
    
    if model.score(data.iloc[val_idx],target['surface'][val_idx]) > 0.92000:
        hm = 30
        plt.figure(figsize=(7, 10))
        plt.title('Feature Importances')
        plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')
        plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    gc.collect()


# In[67]:


print('Avg Accuracy RF', score / folds.n_splits)


# In[68]:


confusion_matrix(measured,target['surface'])


# ### Confusion Matrix Plot

# In[69]:


# https://www.kaggle.com/artgor/where-do-the-robots-drive

def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):
    cm = confusion_matrix(truth, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', size=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()


# In[70]:


plot_confusion_matrix(target['surface'], measured, le.classes_)


# ### Submission (Part I)

# In[71]:


sub['surface'] = le.inverse_transform(predicted.argmax(axis=1))
sub.to_csv('submission.csv', index=False)
sub.head()


# ### Best Submission

# In[72]:


best_sub = pd.read_csv('../input/robots-best-submission/final_submission.csv')
best_sub.to_csv('best_submission.csv', index=False)
best_sub.head(10)


# ## References
# 
# [1] https://www.kaggle.com/vanshjatana/help-humanity-by-helping-robots-4e306b
# 
# [2] https://www.kaggle.com/artgor/where-do-the-robots-drive
# 
# [3] https://www.kaggle.com/gpreda/robots-need-help
# 
# [4] https://www.kaggle.com/vanshjatana/help-humanity-by-helping-robots-4e306b by [@vanshjatana](https://www.kaggle.com/vanshjatana)

# # ABOUT Submissions & Leaderboard

# This kernel [distribution hack](https://www.kaggle.com/donkeys/distribution-hack) by [@donkeys](https://www.kaggle.com/donkeys) simply produces 9 output files, one for each target category. 
# I submitted each of these to the competition to see how much of each target type exists in the test set distribution. Results:
# 
# - carpet 0.06
# - concrete 0.16
# - fine concrete 0.09
# - hard tiles 0.06
# - hard tiles large space 0.10
# - soft pvc 0.17
# - soft tiles 0.23
# - tiled 0.03
# - wood 0.06
# 
# Also posted a discussion [thread](https://www.kaggle.com/c/career-con-2019/discussion/85204)
# 
# 

# **by [@ninoko](https://www.kaggle.com/ninoko)**
# 
# I've probed the public leaderboard and this is what I got
# There are much less surfaces like wood or tiled, and much more soft and hard tiles in public leaderboard. This can be issue, why CV and LB results differ strangely.
# 
# ![graph](https://i.imgur.com/DoFc3mW.png)

# **I will analyze my best submissions in order to find something interesting.**
# 
# Please, feel free to optimize this code.

# In[73]:


sub073 = pd.read_csv('../input/robots-best-submission/mybest0.73.csv')
sub072 = pd.read_csv('../input/robots-best-submission/sub_0.72.csv')
sub072_2 = pd.read_csv('../input/robots-best-submission/sub_0.72_2.csv')
sub071 = pd.read_csv('../input/robots-best-submission/sub_0.71.csv')
sub06 = pd.read_csv('../input/robots-best-submission/sub_0.6.csv')

sub073 = sub073.rename(columns = {'surface':'surface073'})
sub072 = sub072.rename(columns = {'surface':'surface072'})
sub072_2 = sub072_2.rename(columns = {'surface':'surface072_2'})
sub071 = sub071.rename(columns = {'surface':'surface071'})
sub06 = sub06.rename(columns = {'surface':'surface06'})
print ("Submission data is ready")


# In[74]:


sub073.head()


# In[75]:


subtest = pd.concat([sub073['series_id'], sub073['surface073'], sub072['surface072'], sub071['surface071'], sub06['surface06']], axis=1)
subtest.head()


# In[76]:


differents = []
for i in range (0,subtest.shape[0]): 
    labels = list(subtest.iloc[i,1:])
    result = len(set(labels))>1
    if result:
        differents.append((i, str(labels)))
        
differents = pd.DataFrame(differents, columns=['idx','group']) 
differents.head()


# For example the serie with **series_id = 2** has the following predicition:
# 
# ```
# ['tiled', 'tiled', 'tiled', 'fine_concrete']
# ```
# 
# This means that my best submissions (*0.73, 0.72 and  0.71 LB* ) predicted the same: **tiled**, but a worst submission (*0.6 LB*) would have predicted **fine_concrete**.
# 
# ---
# 
# ### So... Why is this interesting?
# 
# In order to improve our classification, LB is indicating us wich kind of surfaces are confused with others.
# In that example, ```tiled``` and ```fine_concrete``` are being **confused** (maybe because the two surfaces are **alike**)
# 
# ---
# 
# As you can see bellow, we have **177 cases of confusion**
# I'm going to plot the tp 10% and see what happens.

# In[77]:


differents['group'].nunique()


# In[78]:


differents['count'] = differents.groupby('group')['group'].transform('count')
differents = differents.sort_values(by=['count'], ascending=False)
differents = differents.drop(['idx'],axis=1)
differents = differents.drop_duplicates()


# In[79]:


differents.head(10)


# We can see that **wood** and **fine_concrete** are really hard to guess.

# ### Maybe this is the most interesting part, the difference between a 0.73 and 0.72 submission.

# In[80]:


differents.tail(10)


# Remember the order at the array is [0.73LB, 0.72LB, 0.71LB, 06LB].
# Series with ```series_id```= 575, 1024, 911, 723, 148, 338 are really interesting because they show important differences between surfaces that often are being confused.

# ## Next Step ??
# 
# - I will create a test dataset with those special cases and then I will ad a new CV stage where I will try to classify those surfaces correctly.
# - I will look for surfaces distinctive features.

# ## Generate a new train and test: Fast Fourier Transform Denoising

# In[81]:


from numpy.fft import *
import matplotlib.pyplot as plt
import matplotlib.style as style 
style.use('ggplot')


# In[82]:


X_train = pd.read_csv('../input/career-con-2019/X_train.csv')
X_test = pd.read_csv('../input/career-con-2019/X_test.csv')
target = pd.read_csv('../input/career-con-2019/y_train.csv')


# In[83]:


series_dict = {}
for series in (X_train['series_id'].unique()):
    series_dict[series] = X_train[X_train['series_id'] == series] 


# In[84]:


# From: Code Snippet For Visualizing Series Id by @shaz13
def plotSeries(series_id):
    style.use('ggplot')
    plt.figure(figsize=(28, 16))
    print(target[target['series_id'] == series_id]['surface'].values[0].title())
    for i, col in enumerate(series_dict[series_id].columns[3:]):
        if col.startswith("o"):
            color = 'red'
        elif col.startswith("a"):
            color = 'green'
        else:
            color = 'blue'
        if i >= 7:
            i+=1
        plt.subplot(3, 4, i + 1)
        plt.plot(series_dict[series_id][col], color=color, linewidth=3)
        plt.title(col)


# In[85]:


plotSeries(1)


# In[86]:


# from @theoviel at https://www.kaggle.com/theoviel/fast-fourier-transform-denoising
def filter_signal(signal, threshold=1e3):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


# Let's denoise train and test angular_velocity and linear_acceleration data

# In[87]:


X_train_denoised = X_train.copy()
X_test_denoised = X_test.copy()

# train
for col in X_train.columns:
    if col[0:3] == 'ang' or col[0:3] == 'lin':
        # Apply filter_signal function to the data in each series
        denoised_data = X_train.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))
        
        # Assign the denoised data back to X_train
        list_denoised_data = []
        for arr in denoised_data:
            for val in arr:
                list_denoised_data.append(val)
                
        X_train_denoised[col] = list_denoised_data
        
# test
for col in X_test.columns:
    if col[0:3] == 'ang' or col[0:3] == 'lin':
        # Apply filter_signal function to the data in each series
        denoised_data = X_test.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))
        
        # Assign the denoised data back to X_train
        list_denoised_data = []
        for arr in denoised_data:
            for val in arr:
                list_denoised_data.append(val)
                
        X_test_denoised[col] = list_denoised_data


# In[88]:


series_dict = {}
for series in (X_train_denoised['series_id'].unique()):
    series_dict[series] = X_train_denoised[X_train_denoised['series_id'] == series] 


# In[89]:


plotSeries(1)


# In[90]:


plt.figure(figsize=(24, 8))
plt.title('linear_acceleration_X')
plt.plot(X_train.angular_velocity_Z[128:256], label="original");
plt.plot(X_train_denoised.angular_velocity_Z[128:256], label="denoised");
plt.legend()
plt.show()


# **Generate new denoised train and test**

# In[91]:


X_train_denoised.head()


# In[92]:


X_test_denoised.head()


# In[93]:


X_train_denoised.to_csv('test_denoised.csv', index=False)
X_test_denoised.to_csv('train_denoised.csv', index=False)

