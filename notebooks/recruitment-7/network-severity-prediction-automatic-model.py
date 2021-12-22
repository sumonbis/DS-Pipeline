
# coding: utf-8

# ### **Loading Libraries**

# In[ ]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import os
print(os.listdir("../input"))
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.metrics import accuracy_score


# ## **Importing datasets**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'event_type=pd.read_csv("../input/event_type.csv",error_bad_lines=False)\ntrain = pd.read_csv("../input/train.csv")\nseverity_type = pd.read_csv("../input/severity_type.csv")\nlog_feature = pd.read_csv("../input/log_feature.csv")\ntest = pd.read_csv("../input/test.csv")\nresource_type = pd.read_csv("../input/resource_type.csv",error_bad_lines=False)\nsample_submission = pd.read_csv("../input/sample_submission.csv")')


# In[ ]:


print("test",test.shape)
print("train",train.shape)


# ### Input datasets heads

# In[ ]:


print('test',test.head())
print('train',train.head(4))
print('sample_submission',sample_submission.head())
print('event_type',event_type.shape,event_type.head(2))
print('severity_type',severity_type.shape,severity_type.head(2))
print('log_feature',log_feature.shape,log_feature.head(2))
print('resource_type',resource_type.shape,resource_type.head(2))


# ### **Visualization of Severity**

# In[ ]:


val=list(train['fault_severity'].value_counts())
for i in range(len(val)):
    print(train['fault_severity'].value_counts().index[i],round(val[i]/sum(val)*100),'%')


# ## ** Data conversion**

# In[ ]:


event_type['id']=pd.to_numeric(event_type['id'],errors='coerce')
#converting object datatype into numeric


# In[ ]:


event_type.dtypes


# # **Training Preprocessing**

# ### **Merging**

# In[ ]:


def merge_fn(df1,df2,col_name,how_param):
    merged_df=df1.merge(df2,how=how_param,on=col_name)
    return merged_df
    


# In[ ]:


train_merge1=merge_fn(train,event_type.drop_duplicates(subset=['id']),'id','left')
train_merge2=merge_fn(train_merge1,severity_type.drop_duplicates(subset=['id']),'id','left')
train_merge3=merge_fn(train_merge2,log_feature.drop_duplicates(subset=['id']),'id','left')
train_merge4=merge_fn(train_merge3,resource_type.drop_duplicates(subset=['id']),'id','left')


# In[ ]:


train_merge4.shape


# In[ ]:


train_merge4.head()


# #### **Calculating mean volumn**

# In[ ]:


train_merge4['mean_volumn']=train_merge4.groupby(['location','event_type','severity_type','log_feature','resource_type'])['volume'].transform('mean')


# #### **Merged Training data**

# In[ ]:


train_merge4.head()


# In[ ]:


train_merge4.dtypes


# ### **Checking for missing values**

# In[ ]:


train_merge4.isnull().sum()


# ### **Finding categorical columns**

# In[ ]:


cat_col=list(set(train_merge4.columns)-set(train_merge4._get_numeric_data().columns))


# ### **Categorical conversion**

# In[ ]:


def categorical_conversion(df,cat_col):
    for i in range(len(cat_col)):
        df[cat_col[i]]=df[cat_col[i]].astype('category')
    return df
    
    


# In[ ]:


train_merge4=categorical_conversion(train_merge4,cat_col)   


# In[ ]:


train_merge4.dtypes


# ### **Label encoding**

# In[ ]:


def label_encoding_conversion(df,cat_col):
    le=preprocessing.LabelEncoder()
    for i in range(len(cat_col)):
        df[cat_col[i]]=le.fit_transform(df[cat_col[i]])
    return df


# In[ ]:


train_merge4.columns


# In[ ]:


train_merge4=label_encoding_conversion(train_merge4,cat_col)


# In[ ]:


train_merge4.columns


# ### **Droping unique values**

# In[ ]:


train_merge4.drop(['id'],axis=1,inplace=True)


# In[ ]:


target=train_merge4[['fault_severity']]


# In[ ]:


train_merge4.drop(['fault_severity'],axis=1,inplace=True)


# In[ ]:


train_merge4.head()


# In[ ]:


train_merge4.dtypes


# In[ ]:


test.head()


# ## **TEST data preparation**

# In[ ]:


test.head()


# In[ ]:


test.shape


# ## ** Test data merging**

# In[ ]:


test_merge1=merge_fn(test,event_type.drop_duplicates(subset=['id']),'id','left')
test_merge2=merge_fn(test_merge1,severity_type.drop_duplicates(subset=['id']),'id','left')
test_merge3=merge_fn(test_merge2,log_feature.drop_duplicates(subset=['id']),'id','left')
test_merge4=merge_fn(test_merge3,resource_type.drop_duplicates(subset=['id']),'id','left')


# In[ ]:


test_merge4.shape


# ###**Adding new feature- Mean volume**

# In[ ]:


test_merge4['mean_volumn']=test_merge4.groupby(['location','event_type','severity_type','log_feature','resource_type'])['volume'].transform('mean')


# In[ ]:


severity_type.head()


# In[ ]:


test_merge4.head(2)


# #### ** Categorical columns**

# In[ ]:


cat_col


# ### **Categorical conversion **

# In[ ]:


test_merge4=categorical_conversion(test_merge4,cat_col)  


# In[ ]:


test_merge4.dtypes


# ### **Label encoding**

# In[ ]:


test_merge4=label_encoding_conversion(test_merge4,cat_col)


# In[ ]:


test_merge4.dtypes


# ### **Removing  unique columns**

# In[ ]:


test_merge4.drop(['id'],axis=1,inplace=True)


# In[ ]:


train_merge4.columns


# In[ ]:


test_merge4.columns


# ### **LogisticRegression**

# In[ ]:


train_merge4.columns


# In[ ]:


lr=LogisticRegression()
lr.fit(train_merge4,target)
lr_pred=lr.predict(test_merge4)
accuracy_score(pd.DataFrame(lr.predict(train_merge4)),target)


# ### **RandomForestClassifier**

# In[ ]:


rf=RandomForestClassifier()
rf.fit(train_merge4,target)
rf_pred=rf.predict(test_merge4)
accuracy_score(pd.DataFrame(rf.predict(train_merge4)),target)


# ### **GaussianNB**

# In[ ]:



nb=GaussianNB()
nb.fit(train_merge4,target)
nb.predict(test_merge4)
accuracy_score(pd.DataFrame(nb.predict(train_merge4)),target)


# ### **DecisionTreeClassifier**

# In[ ]:



dt=tree.DecisionTreeClassifier()
dt.fit(train_merge4,target)
dt.predict(test_merge4)
accuracy_score(pd.DataFrame(dt.predict(train_merge4)),target)


# ### **SVC**

# In[ ]:



svc_ml=svm.SVC()
svc_ml.fit(train_merge4,target)
svc_ml.predict(test_merge4)
accuracy_score(pd.DataFrame(svc_ml.predict(train_merge4)),target)


# ### **AdaBoostClassifier**

# In[ ]:



ada=AdaBoostClassifier()
ada.fit(train_merge4,target)
ada.predict(test_merge4)
accuracy_score(pd.DataFrame(ada.predict(train_merge4)),target)


# ### **KNeighborsClassifier**

# In[ ]:



knn=KNeighborsClassifier()
knn.fit(train_merge4,target)
knn.predict(test_merge4)
accuracy_score(pd.DataFrame(knn.predict(train_merge4)),target)


# ### **GradientBoostingClassifier**

# In[ ]:



gb=ensemble.GradientBoostingClassifier()
gb.fit(train_merge4,target)
gb_pre=gb.predict(test_merge4)
accuracy_score(pd.DataFrame(gb.predict(train_merge4)),target)


# ## Model comparison consolidate function

# In[ ]:


dic_data={}
list1=[]
max_clf_output=[]
tuple_l=()
def data_modeling(X,target,model):
    for i in range(len(model)):
        ml=model[i]
        ml.fit(X,target)
        pred=ml.predict(X)
        acc_score=accuracy_score(pd.DataFrame(ml.predict(X)),target)
        tuple_l=(ml.__class__.__name__,acc_score)
        dic_data[ml.__class__.__name__]=[acc_score,ml]
        list1.append(tuple_l)
        print(dic_data)
    for name,val in dic_data.items():
        if val==max(dic_data.values()):
            max_lis=[name,val]
            print('Maximum classifier',name,val)

    return list1,max_lis

list1,max_lis=data_modeling(train_merge4,target,[AdaBoostClassifier(),KNeighborsClassifier(),
svm.SVC(),RandomForestClassifier(),
tree.DecisionTreeClassifier(),
GaussianNB(),
LogisticRegression(),
ensemble.GradientBoostingClassifier()])


# In[ ]:


model=max_lis[1][1]


# ## **Model score Visualization**

# In[ ]:


modelscore_df=pd.DataFrame(list1,columns=['Classifier',"Accuracy score"])


# In[ ]:


modelscore_df


# In[ ]:


modelscore_df['classifier code']=np.arange(8)


# In[ ]:


modelscore_df


# In[ ]:


modelscore_df.shape[0]


# ### ** Classifier selection **

# In[ ]:


clf_sel=modelscore_df.iloc[modelscore_df['Accuracy score'].idxmax()]
clf_name=clf_sel[0]


# In[ ]:


modelscore_df.plot.bar(x='classifier code', y='Accuracy score', rot=0)


# ### **Submission file generation**

# In[ ]:



predict_test=rf.predict_proba(test_merge4)
pred_df=pd.DataFrame(predict_test,columns=['predict_0', 'predict_1', 'predict_2'])
submission=pd.concat([test[['id']],pred_df],axis=1)
submission.to_csv('sub.csv',index=False,header=True)


# In[ ]:


submission

