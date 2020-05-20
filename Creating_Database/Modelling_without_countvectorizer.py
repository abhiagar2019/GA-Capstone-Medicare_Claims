#!/usr/bin/env python
# coding: utf-8

# In[56]:


#import requests
#import bs4
#from bs4 import BeautifulSoup
#from tqdm import tqdm

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
#import seaborn as sns

#sns.set(font_scale=1.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

#import scikitplot as skplt
from matplotlib.colors import ListedColormap
#cmap = ListedColormap(sns.color_palette("husl", 3))

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor  

#sns.set(font_scale=1.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
import scipy.stats as stats
from sklearn.tree import DecisionTreeRegressor
from six import StringIO
from sklearn.tree import export_graphviz
#import pydotplus

#import scikitplot as skplt

plt.rc("figure", figsize=(9, 7))
#sns.set(font_scale=1.5)

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

warnings.simplefilter('ignore')

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor

pd.set_option('display.max_columns', 500)   # to display 500 columns
pd.set_option('display.max_rows', 500) # to display 500 rows

import time
from datetime import datetime


# In[ ]:


'''
import psycopg2

# DSN (data source name) format for database connections:  
# [protocol / database  name]://[username]:[password]@[hostname / ip]:[port]/[database name here]


# on your computer you are the user postgres (full administrative access)
db_user = 'postgres'
# if you need a password to access a database, put it here
db_password = ''
# on your computer, use localhost
db_host = 'localhost'
# the default port for postgres is 5432
db_port = 5432
# we want to connect to the northwind database
database =   'cms_claims' # 'cms_medicare_claims'  #

conn_str = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{database}'
conn = psycopg2.connect(conn_str)
'''


# In[ ]:


# scp -i "abhi.pem" cms.csv ubuntu@ec2-35-171-193-243.compute-1.amazonaws.com:.  (run from terminal)


# In[ ]:





# In[14]:


'''
import pandas as pd
import boto3
import S3Connection

bucket = "cmsclaims"
file_name = "cms.csv"

conn = S3Connection()
my_bucket = conn.get_bucket(bucket, validate=False)
    


s3 = boto3.client('s3') 
# 's3' is a key word. create connection to S3 using default config and all buckets within S3

obj = s3.get_object(Bucket= bucket, Key= file_name) 
# get object and file (key) from bucket

df = pd.read_csv(obj['Body']) # 'Body' is a key word

import botocore
import boto3

session = boto3.session.Session(aws_access_key_id='AKIAIS2Y27XS72YJY7VA') #, aws_secret_access_key=<>, region_name=<>)


client = boto3.client('s3') #low-level functional API

s3 = boto3.resource('s3')
bucket = s3.Bucket('cmsclaims')


KEY = 'cms.csv'
try:
    bucket.download_file(KEY,'cms.csv')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise
        '''


# In[15]:


df = pd.read_csv('cms.csv') #, header=False, inferSchema=True)


# In[16]:


print(df.shape)
df.head(2)


# In[3]:


'''
# Check Tables in the Database
query = """
SELECT tablename 
FROM pg_catalog.pg_tables 
WHERE schemaname='public'
"""

pd.read_sql(query, con=conn)
'''


# In[10]:


query = """

SELECT *
FROM final2008_2009
"""

df = pd.read_sql(query, conn) 
# df is patients who are in 2008 and 2009 benefciiary, unfortunately 2010 doesnt have common benefciary patients

print(df.shape)
df.head(2)


# In[17]:


df.isnull().sum()


# In[12]:


# Replacing  Nulls with 0

query = """

SELECT 

     "DESYNPUF_ID"
    , "BENE_BIRTH_DT"
    , COALESCE("BENE_DEATH_DT", '2100-01-01') as "BENE_DEATH_DT"
    , COALESCE("BENE_SEX_IDENT_CD", 0) as "BENE_SEX_IDENT_CD"
    , COALESCE("BENE_RACE_CD", 0) as "BENE_RACE_CD"
    , cast((CASE when "END_STAGE_RENAL2008" like 'Y' then '1' else '0' END) as int) as "END_STAGE_RENAL2008"
    , cast((CASE when "END_STAGE_RENAL2009" like 'Y' then '1' else '0' END) as int) as "END_STAGE_RENAL2009"
    , COALESCE("SP_ALZHDMTA2008", 0) as "SP_ALZHDMTA2008"
    , COALESCE("SP_ALZHDMTA2009", 0) as "SP_ALZHDMTA2009"
    , COALESCE("SP_CHF2008", 0) as "SP_CHF2008"
    , COALESCE("SP_CHF2009", 0) as "SP_CHF2009"
    , COALESCE("SP_CHRNKIDN2008", 0) as "SP_CHRNKIDN2008"
    , COALESCE("SP_CHRNKIDN2009", 0) as "SP_CHRNKIDN2009"
    , COALESCE("SP_CNCR2008", 0) as "SP_CNCR2008"
    , COALESCE("SP_CNCR2009", 0) as "SP_CNCR2009"
    , COALESCE("SP_COPD2008", 0) as "SP_COPD2008"
    , COALESCE("SP_COPD2009", 0) as "SP_COPD2009"
    , COALESCE("SP_DEPRESSN2008", 0) as "SP_DEPRESSN2008"
    , COALESCE("SP_DEPRESSN2009", 0) as "SP_DEPRESSN2009"
    , COALESCE("SP_DIABETES2008", 0) as "SP_DIABETES2008"
    , COALESCE("SP_DIABETES2009", 0) as "SP_DIABETES2009"
    , COALESCE("SP_ISCHMCHT2008", 0) as "SP_ISCHMCHT2008"
    , COALESCE("SP_ISCHMCHT2009", 0) as "SP_ISCHMCHT2009"
    , COALESCE("SP_OSTEOPRS2008", 0) as "SP_OSTEOPRS2008"
    , COALESCE("SP_OSTEOPRS2009", 0) as "SP_OSTEOPRS2009"
    , COALESCE("SP_RA_OA2008", 0) as "SP_RA_OA2008"
    , COALESCE("SP_RA_OA2009", 0) as "SP_RA_OA2009"
    , COALESCE("SP_STRKETIA2008", 0) as "SP_STRKETIA2008"
    , COALESCE("TOTAL_DIAGNOSIS2008", 0) as "TOTAL_DIAGNOSIS2008"
    , COALESCE("TOTAL_DIAGNOSIS2009", 0) as "TOTAL_DIAGNOSIS2009"
    , COALESCE("TOTAL_PAYMENT2008", 0) as "TOTAL_PAYMENT2008"
    , COALESCE("TOTAL_PAYMENT2009", 0) as "TOTAL_PAYMENT2009"
    , COALESCE("CHANGE_IN_PAYMEMT", 0) as "CHANGE_IN_PAYMEMT"
    , COALESCE("CHANGE_IN_DIANOSIS", 0) as "CHANGE_IN_DIAGNOSIS"
    , COALESCE("DEAD", 0) as "DEAD"
    , COALESCE("AGE", 0) as "AGE"
    , COALESCE("LOS", 0) as "LOS"
    , COALESCE("NUM_INPT_ADM", 0) as "NUM_INPT_ADM"
    , COALESCE("TOTAL_INPT_COST", 0) as "TOTAL_INPT_COST"
    , COALESCE("TOTAL_INPT_DIAGNOSIS", 0) as "TOTAL_INPT_DIAGNOSIS"
    , COALESCE("TOTAL_INPT_PROCS", 0) as "TOTAL_INPT_PROCS"
    , COALESCE("READMIT7", 0) as "READMIT7" 
    , COALESCE("READMIT30", 0) as "READMIT30"
    , COALESCE("READMIT60", 0) as "READMIT60"
    , COALESCE("READMIT90", 0) as "READMIT90"
    , COALESCE("NUM_OPD_VISIT", 0) as "NUM_OPD_VISIT"
    , COALESCE("TOTAL_OPD_COST", 0) as "TOTAL_OPD_COST"
    , COALESCE("TOTAL_OPD_DIAGNOSIS", 0) as "TOTAL_OPD_DIAGNOSIS"
    , COALESCE("TOTAL_OPD_PROCS", 0) as "TOTAL_OPD_PROCS"
    , COALESCE("TOTAL_OPD_HCPCS", 0) as "TOTAL_OPD_HCPCS"
    , COALESCE("TOTAL_QTY_DSPNSD_NUM", 0) as "TOTAL_QTY_DSPNSD_NUM"
    , COALESCE("PTNT_PAY_RX_AMT", 0) as "PTNT_PAY_RX_AMT"
    , COALESCE("TOT_RX_CST_AMT", 0) as "TOT_RX_CST_AMT"
    , COALESCE("INPT_DIAGS", '') as "INPT_DIAGS"
    , COALESCE("INPT_PROCS", '') as "INPT_PROCS"
    , COALESCE("OPD_DIAGS", '') as "OPD_DIAGS"
    , COALESCE("OPD_PROCS", '') as "OPD_PROCS"
    , COALESCE("OPD_HCPCS", '') as "OPD_HCPCS"
    , CASE when "READMIT7" = 0.0 then 0 else 1 END as "READMIT7_FLAG"  
    , CASE when "READMIT30" = 0.0 then 0 else 1 END as "READMIT30_FLAG" 
    , CASE when "READMIT60" = 0.0 then 0 else 1 END as "READMIT60_FLAG" 
    , CASE when "READMIT90" = 0.0 then 0 else 1 END as "READMIT90_FLAG" 

FROM final2008_2009
"""

df = pd.read_sql(query, conn) 
# df is patients who are in 2008 and 2009 benefciiary, unfortunately 2010 doesnt have common benefciary patients

print(df.shape)
df.head(2)


# In[13]:


# Checking for nulls

df.isnull().sum()


# In[104]:


# write to csv

#df.to_csv(r'/Users/abhi/Documents/Abhi/General Assembly/Immersive course/Course/DSI12-lessons/projects/cms.csv', header=True) 


# In[103]:


#Checking data types

#df.dtypes
#df.columns


# #### Visualizing the spread of information 

# In[34]:


df.iloc[:,5:24].hist(figsize=(24,16), sharex=True, sharey=True)
plt.show()  # 1 means have diesease, 2 means no disease


# In[35]:


df[['TOTAL_INPT_COST', 'TOTAL_OPD_COST', 'TOTAL_PAYMENT2008',
    'TOTAL_PAYMENT2009']].hist(figsize=(12,8), sharex=True, sharey=True)
plt.show()  


# In[30]:





# In[36]:


df[['TOTAL_INPT_DIAGNOSIS','TOTAL_INPT_PROCS', 'TOTAL_OPD_DIAGNOSIS','TOTAL_OPD_PROCS',
                                   'TOTAL_OPD_HCPCS',]].hist(figsize=(12,20), sharex=True, sharey=True) #bins=10
plt.show()  #  'NUM_OPD_VISIT'


# In[37]:


df[['READMIT7','READMIT30','READMIT60','READMIT90']].hist(figsize=(12,8), sharex=True, sharey=True)
plt.show()


# In[38]:


df[['DEAD']].hist(figsize=(12,8), sharex=True, sharey=True)
plt.show()


# ### Looking for corelation between the variables

# In[16]:


corr = df.corr()
corr


# In[17]:


fig, ax = plt.subplots(figsize=(60,36))
matrix = np.triu(corr)
ax = sns.heatmap(corr, annot = True, mask=matrix, center= 0, cmap="BuPu" ,fmt='.1g'); #cmap= 'coolwarm',    


# #### Correlation of predcitor variable against the target variables

# In[18]:


corr1 = df.corr()['READMIT30_FLAG'][:] # 'READMIT7', 'READMIT30', 'READMIT60', 'READMIT90'
corr_target = corr1.sort_values(ascending=False)
corr_target


# In[19]:


corr_target_df=pd.DataFrame(corr_target)
fig, ax = plt.subplots(figsize=(8,10 ))
sns.heatmap(corr_target_df);


# ### Feature selection (excuding string summation of ICDs and features which are showing colinearity etc.)

# In[20]:


datetime.now()


# In[32]:


df_selected = df[['BENE_SEX_IDENT_CD',
       'BENE_RACE_CD', 'END_STAGE_RENAL2008', 'END_STAGE_RENAL2009',
       'SP_ALZHDMTA2008', 'SP_ALZHDMTA2009', 'SP_CHF2008', 'SP_CHF2009',
       'SP_CHRNKIDN2008', 'SP_CHRNKIDN2009', 'SP_CNCR2008', 'SP_CNCR2009',
       'SP_COPD2008', 'SP_COPD2009', 'SP_DEPRESSN2008', 'SP_DEPRESSN2009',
       'SP_DIABETES2008', 'SP_DIABETES2009', 'SP_ISCHMCHT2008',
       'SP_ISCHMCHT2009', 'SP_OSTEOPRS2008', 'SP_OSTEOPRS2009', 'SP_RA_OA2008',
       'SP_RA_OA2009', 'SP_STRKETIA2008', 'TOTAL_DIAGNOSIS2008',
       'TOTAL_DIAGNOSIS2009', 'TOTAL_PAYMENT2008', 'TOTAL_PAYMENT2009',
       'CHANGE_IN_PAYMEMT', 'CHANGE_IN_DIAGNOSIS', 'AGE', 'LOS',
       'NUM_INPT_ADM', 'TOTAL_INPT_COST', 'TOTAL_INPT_DIAGNOSIS',
       'TOTAL_INPT_PROCS', 
       'NUM_OPD_VISIT', 'TOTAL_OPD_COST', 'TOTAL_OPD_DIAGNOSIS',
       'TOTAL_OPD_PROCS', 'TOTAL_OPD_HCPCS', 'TOTAL_QTY_DSPNSD_NUM',
       'PTNT_PAY_RX_AMT', 'TOT_RX_CST_AMT', 'READMIT30_FLAG']].copy()   # , 'DEAD'
print(df_selected.shape)
df_selected.head(2)  
# 'READMIT7', 'READMIT30', 'READMIT60', 'READMIT90',


# In[28]:


# Check if and who was readmiited 8 times


# ### Selecting Predictor and Target Variables

# In[33]:


y = df_selected.pop('READMIT30_FLAG')  #df_selected.pop('DEAD') 
X = df_selected  

print(X.shape)
print(y.shape)


# ### Checking for baseline accuracy

# In[34]:


y.value_counts(normalize=True)


# #### Since the dataset doesn't seem to be too imbalance, over or under sampling isn't required

# ### Train - Test split

# In[35]:


# Creating Train, test split with Target variable for fixed features 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y ) #, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# #### Standardize the data

# In[36]:


# Standardize the data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# ### Principal Component Analysis to select important features and remove any colinear features

# In[37]:


datetime.now()


# In[38]:


from sklearn.decomposition import PCA

pca = PCA(n_components=20)

# PCA fit-tranform on train data
X_train_pca = pca.fit_transform(X_train)

# Converting into DataFrame
PCA_col_dict = {'PCA'+str(i+1): pca.components_[i]
            for i in range(len(pca.components_))}
X_train_pca = pd.DataFrame(X_train_pca, columns=PCA_col_dict) 
print(X_train_pca.shape)

# PCA tranform on test data
X_test_pca = pca.transform(X_test)
X_test_pca = pd.DataFrame(X_test_pca, columns=PCA_col_dict) 

print(X_test_pca.shape)

X_train_pca.head(2)


# In[40]:


explained_variance = pca.explained_variance_ratio_
explained_variance.cumsum()


# In[44]:


fig, ax = plt.subplots(figsize=(8, 6))
x_values = list(range(1, pca.n_components_+1))
ax.plot(x_values, explained_variance, lw=2)
ax.scatter(x_values, explained_variance, s=120)
ax.plot(x_values, np.cumsum(explained_variance), lw=2)
ax.scatter(x_values, np.cumsum(explained_variance), s=120)
#ax.plot(x_values,0.9)
ax.set_title('Explained variance pct\n', fontsize=20)
ax.set_xlabel('Principal Component', fontsize=16)
ax.set_ylabel('Variance Explained (%)', fontsize=16)
plt.show()


# #### We select 32 PCA's which return 95% of variance  (28 PCA return 90% variance)

# In[45]:


X_train_pca_corr = X_train_pca.corr()

mask = np.zeros_like(X_train_pca_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(24, 16))
ax = sns.heatmap(X_train_pca_corr, mask=mask, annot=True)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), rotation=45)
ax.set_ylim(X_train_pca_corr.shape[1], 0)
plt.show()


# In[46]:


X_test_pca_corr = X_test_pca.corr()

mask = np.zeros_like(X_test_pca_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(24, 16))
ax = sns.heatmap(X_test_pca_corr, mask=mask, annot=True)
ax.set_xticklabels(ax.xaxis.get_ticklabels(), rotation=45)
ax.set_ylim(X_test_pca_corr.shape[1], 0)
plt.show()


# ### Building basic Logistic Regression model for readmission prediction

# In[55]:


datetime.now()


# In[56]:


modelLR = LogisticRegression()
modelLR.fit(X_train_pca,y_train)
datetime.now()


# In[57]:


print("Score(Train): ", modelLR.score(X_train_pca, y_train))
print("Cross Val Score (Train): ", cross_val_score(modelLR, X_train_pca, y_train, cv =5).mean()) 
print("Score (Test): ", modelLR.score(X_test_pca, y_test))


# In[58]:


datetime.now()


# In[59]:


importance = modelLR.coef_[0]

importanceDF = pd.DataFrame(importance, index=X_train_pca.columns, columns=["Importance"])
importanceDF.sort_values(by='Importance', ascending=False).head(15)


# In[60]:


importanceDF.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(8, 6))
plt.show()


# In[61]:


#print(pca.inverse_transform(X_train_pca)) 


# In[ ]:





# In[62]:


# Predictons and Probabilities

predictions_LRBase = modelLR.predict(X_test_pca)
#predictions_LRBase

probabilities_LRBase = modelLR.predict_proba(X_test_pca)
#probabilities_LRBase


# In[63]:


labels=[0, 1]
confusion_mat = confusion_matrix(y_test, predictions_LRBase, labels=labels) # it is imp to put labels else its confusing
#print(labels)
#print(confusion_mat)

pd.DataFrame(confusion_mat,
             columns=['Predicted_No_Readmissions', 'Predicted_Readmissions'],
             index=['True_No_Readmissions', 'True_Readmissions'])


# In[64]:


print(classification_report(y_test, predictions_LRBase))


# In[53]:


def plot_f1_lines(figsize=(8,6),fontsize=16):
    '''Create f1-score level lines to be added to the precison-recall plot'''

    fig, ax = plt.subplots(figsize=figsize)
    
    # add lines of constant F1 scores
    
    for const in np.linspace(0.2,0.9,8):
        x_vals = np.linspace(0.001, 0.999, 100)
        y_vals = 1./(2./const-1./x_vals)
        ax.plot(x_vals[y_vals > 0], y_vals[y_vals > 0],
                 color='lightblue', ls='--', alpha=0.9)
        ax.set_ylim([0, 1])
        ax.annotate('f1={0:0.1f}'.format(const),
                     xy=(x_vals[-10], y_vals[-2]+0.0), fontsize=fontsize)

    return fig, ax


# In[66]:


# plot_f1_lines # already defined

# Recall Precision plot

fig, ax = plot_f1_lines()
skplt.metrics.plot_precision_recall(y_test, probabilities_LRBase, 
                       plot_micro=True, 
                       title_fontsize=20, text_fontsize=16, cmap=cmap, ax=ax)
ax.legend(loc=[1.1,0])
plt.show()


# In[67]:


# ROC plot

skplt.metrics.plot_roc(y_test, probabilities_LRBase, plot_micro=True, plot_macro=True, 
                       title_fontsize=20, text_fontsize=16, figsize=(8,6), cmap=cmap)
plt.show()


# In[60]:


# Logistic Regression GridSearch

gs_lr_params = {'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'C': np.logspace(-3, 0, 50) } #np.logspace(-3, 0, 100)}

model_LR = LogisticRegression(solver='liblinear', multi_class='ovr')

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

lr_gridsearch = GridSearchCV(estimator=model_LR,
                             param_grid=gs_lr_params,
                             cv=kf,
                             scoring='accuracy',
                             n_jobs=3,     # runs on 3 cores
                             verbose=1)


# In[61]:


datetime.now()  


# In[ ]:


lr_gridsearch.fit(X_train, y_train)


# In[ ]:


print('Best Paramaters (logistic Regression): ', lr_gridsearch.best_params_)
print('Best Score (logistic Regression): ', lr_gridsearch.best_score_)
print('Best Logistic Regression: ' , lr_gridsearch.best_estimator_)


# In[ ]:


datetime.now()


# In[ ]:


lr_gridsearch.score(X_test, y_test)


# In[ ]:


datetime.now()


# In[ ]:


# Decision Tree


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Random Forest

# In[72]:


datetime.now()


# In[73]:


modelRF = RandomForestClassifier() 
modelRF.fit(X_train_pca,y_train)


# In[ ]:


datetime.now()


# In[74]:


print("Score(Train): ", modelRF.score(X_train_pca, y_train))
print("Cross Val Score (Train): ", cross_val_score(modelRF, X_train_pca, y_train, cv =5).mean()) 
print("Score (Test): ", modelRF.score(X_test_pca, y_test))


# In[75]:


datetime.now()


# In[ ]:


''' # when ran withoOUT vectorizer - RF (but WITH gini and 100 estimators)

Score(Train):  1.0
Cross Val Score (Train):  0.9432718529372724
Score (Test):  0.9421599441243234
    
# when ran withoOUT vectorizer - RF (but WITHout gini and 100 estimators)

Score(Train):  0.9972236010276494  (28 PCA)
Cross Val Score (Train):  0.9416274400747433
Score (Test):  0.9416141208299696
    
# when ran WITH vectorizer - RF  (but without gini and 100 estimators)

Score(Train):  0.9970763264259727
Cross Val Score (Train):  0.8548963139654087
Score (Test):  0.858278969301595
    
# when ran WITH vectorizer - RF  (but WITH gini and 100 estimators)

Score(Train):  1
Cross Val Score (Train):  
Score (Test): 
'''


# In[ ]:


datetime.now()


# ### Random Forest Grid Search

# In[ ]:


datetime.now() # took 16 mins on my machine, fran in 3.5 mins on AWS


# In[ ]:


param_grid_RF = { 
                    'n_estimators': [10, 15, 50, 100, 200],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth' : [4,7,10, 20, None],
                    'criterion' :['gini', 'entropy']
                }


# In[ ]:


modelRF_ = RandomForestClassifier(random_state=1) 
modelRF_GS = GridSearchCV(estimator=modelRF_, param_grid=param_grid_RF, cv= 3)  # cv=5
modelRF_GS.fit(X_train, y_train)


# In[ ]:


datetime.now()


# In[ ]:


print("Best score (Decision Tree): ", modelRF_GS.best_score_)
print("Best params (Decision Tree): ", modelRF_GS.best_params_)
print("Best estimator (Decision Tree): ", modelRF_GS.best_estimator_)


# In[ ]:


datetime.now()


# In[ ]:


print('Test Score: ' , modelRF_GS.best_estimator_.score(X_test,y_test))


# In[ ]:


datetime.now()


# In[ ]:


predictions_RF_GS = modelRF_GS.best_estimator_.predict(X_test)
#predictions_RF_GS

probabilities_RF_GS = modelRF_GS.best_estimator_.predict_proba(X_test)
#probabilities_RF_GS


# In[ ]:


labels=[0, 1]
confusion_mat = confusion_matrix(y_test, predictions_RF_GS, labels=labels) # it is imp to put labels else its confusing
#print(labels)
#print(confusion_mat)

pd.DataFrame(confusion_mat,
             columns=['Predicted_No_Readmissions', 'Predicted_Readmissions'],
             index=['True_No_Readmissions', 'True_Readmissions'])


# In[ ]:


print(classification_report(y_test, predictions_RF_GS))


# In[54]:


# Recall Precision plot

fig, ax = plot_f1_lines()
skplt.metrics.plot_precision_recall(y_test, probabilities_RF_GS, 
                       plot_micro=True, 
                       title_fontsize=20, text_fontsize=16, cmap=cmap, ax=ax)
ax.legend(loc=[1.1,0])
plt.show()


# In[55]:


# ROC plot

skplt.metrics.plot_roc(y_test, probabilities_RF_GS, plot_micro=True, plot_macro=True, 
                       title_fontsize=20, text_fontsize=16, figsize=(8,6), cmap=cmap)
plt.show()


# In[ ]:


datetime.now()


# ### Decison Tree

# In[ ]:


DT_Classifier_params = {'max_depth': list(range(1, 11))+[None], #'max_depth': [None, 1,2,3,4,5],
                        'criterion' :['gini', 'entropy'], 
                        'max_features': [None, 1, 2, 3],
                        'min_samples_split': [2, 5, 10, 20, 30, 50],
                        'min_samples_leaf': [1, 2, 3, 4]
                        #,'ccp_alpha': [0, 0.001, 0.005, 0.01]
}


# In[ ]:


DT_Classifier_gridsearch = GridSearchCV(estimator=classifier,
                                         param_grid=DT_Classifier_params,
                                         cv=kf,
                                         #scoring='accuracy',
                                         n_jobs=3,     # runs on 3 cores
                                         verbose=1)


# In[ ]:


DT_Classifier_gridsearch.fit(X_train, y_train)


# In[ ]:


datetime.now()


# In[ ]:


print("Best score (Decision Tree): ", DT_Classifier_gridsearch.best_score_)
print("Best params (Decision Tree): ", DT_Classifier_gridsearch.best_params_)
print("Best estimator (Decision Tree): ", DT_Classifier_gridsearch.best_estimator_)


# In[ ]:


datetime.now()


# In[ ]:


print('Test Score: ' , DT_Classifier_gridsearch.best_estimator_.score(X_test,y_test))


# In[ ]:


datetime.now()


# In[ ]:


# Predictons and Probabilities 
predictions_DT = DT_Classifier_gridsearch.best_estimator_.predict(X_test) 
#predictions_DT 
probabilities_DT = DT_Classifier_gridsearch.best_estimator_.predict_proba(X_test) 
#probabilities_DT


# In[ ]:


labels=[0, 1]
confusion_mat = confusion_matrix(y_test, predictions_DT, labels=labels) # it is imp to put labels else its confusing
#print(labels)
#print(confusion_mat)

pd.DataFrame(confusion_mat,
             columns=['Predicted_No_Readmissions', 'Predicted_Readmissions'],
             index=['True_No_Readmissions', 'True_Readmissions'])


# In[ ]:


print(classification_report(y_test, predictions_DT))


# In[ ]:


datetime.now()


# In[ ]:





# ### KNN GridSerach

# In[ ]:


datetime.now()


# In[ ]:


gs_knn_params = {
                'n_neighbors': [5, 15, 25, 35, 40, 45, 50, 60, 75],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
                }

model_KNN = KNeighborsClassifier()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# In[ ]:


knn_gridsearch = GridSearchCV(estimator=model_KNN,
                             param_grid=gs_knn_params,
                             cv=kf,
                             scoring='accuracy',
                             n_jobs=3,     # runs on 3 cores
                             verbose=1) # return_train_score=True
knn_gridsearch.fit(X_train, y_train)


# In[ ]:


datetime.now()


# In[ ]:


# Printing parameters

print('Best Paramaters (KNN): ', knn_gridsearch.best_params_)
print('Best Score (KNN): ', knn_gridsearch.best_score_)
print('Best KNN ' , knn_gridsearch.best_estimator_)


# In[ ]:


datetime.now()


# In[ ]:


print('Test Score: ' , knn_gridsearch.best_estimator_.score(X_test,y_test))


# In[ ]:


datetime.now()


# In[ ]:


predictions_KNN = knn_gridsearch.best_estimator_.predict(X_test)
#predictions_KNN

probabilities_KNN = knn_gridsearch.best_estimator_.predict_proba(X_test)
#probabilities_KNN


# In[ ]:


labels=[0, 1]
confusion_mat = confusion_matrix(y_test, predictions_KNN, labels=labels) # it is imp to put labels else its confusing
#print(labels)
#print(confusion_mat)

pd.DataFrame(confusion_mat,
             columns=['Predicted_No_Readmissions', 'Predicted_Readmissions'],
             index=['True_No_Readmissions', 'True_Readmissions'])


# In[ ]:


print(classification_report(y_test, predictions_KNN))


# In[ ]:


datetime.now()


# In[ ]:





# ### SVM

# In[ ]:


datetime.now()


# In[ ]:


gs_SVM_params = {'gamma': np.linspace(0.01, 2, 10),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # , 'precomputed'
                'C': np.logspace(-2, 2, 11)}   

modelSVM_GS = svm.SVC()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

SVM_gridsearch = GridSearchCV(estimator=modelSVM_GS,
                             param_grid=gs_SVM_params,
                             cv=kf,
                             #scoring='accuracy',
                             n_jobs=3,     # runs on 3 cores
                             verbose=1,
                             error_score = 0.0)


SVM_gridsearch.fit(X_train, y_train)


# In[ ]:


datetime.now()


# In[ ]:


print('Best Paramaters (SVM): ', SVM_gridsearch.best_params_)
print('Best Score (SVM): ', SVM_gridsearch.best_score_)
print('Best SVM: ' , SVM_gridsearch.best_estimator_)


# In[ ]:


datetime.now()


# In[ ]:


print('Test Score: ' , SVM_gridsearch.best_estimator_.score(X_test,y_test))


# In[ ]:


datetime.now()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def grid_search_func(estimator, params, X_train, y_train, X_test, y_test, scoring_function=metrics.accuracy_score, scoring='accuracy', cv=5):
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        return_train_score=True,
        scoring=scoring,
        cv=cv)

    gs.fit(X_train, y_train)
    datetime.now()
    
    print("Best score")
    print(gs.best_score_)
    print()
    print("Best estimator")
    print(gs.best_estimator_.get_params())
    print()
    datetime.now()

    predictions = gs.best_estimator_.predict(X_test)
    print('Test score: ', scoring_function(y_test, predictions))
    print()
    print_cm_cr(y_test, predictions)
    datetime.now()
    
    return gs


# In[ ]:


params_rbf = {'C': np.logspace(-2, 2, 11),
              'gamma': np.linspace(0.01, 2, 10)}

gs_rbf = grid_search_func(model_rbf, params_rbf,
                          X_train, y_train, X_test, y_test)


# In[ ]:


datetime.now()


# In[ ]:





# In[ ]:





# In[ ]:


# Write function for diff models and grid search pipeline
try with and without over sampling
LR,
KNN,
DT,
RF
SVM
get_ipython().run_line_magic('pinfo', 'AdaBoost')
change threshold of probability and see if the accuracy gets better
is recall or precision more imp for me 

death prediction
cost prediction
disease prediction


# In[ ]:


#week 5 day 1

pipe = Pipeline(steps=[('scaler', scaler),
                       ('model', model)])

pipe.fit(X, y)

columns_to_drop = ['PassengerId', 'Name']
columns_to_dummify = ['Sex', 'Pclass', 'Embarked']

tprep = TitanticPreprocessor(columns_to_drop=columns_to_drop,
                             columns_to_dummify=columns_to_dummify)
scaler = StandardScaler()
model = LogisticRegression(solver='lbfgs', random_state=1)

pipe = Pipeline(steps=[('titanic_prep', tprep),
                       ('scaler', scaler),
                       ('model', model)
                      ]
               )


scaler = StandardScaler()
#scaler = MinMaxScaler()
poly = PolynomialFeatures(include_bias=False)
model = LogisticRegression(solver='lbfgs')
#model = KNeighborsClassifier()
fu_pipe = Pipeline(steps=(('union', fu),
                          ('poly', poly),
                          ('scaler', scaler),
                          ('model', model)))


# In[ ]:


# setup the grid search

params = {'C': np.logspace(-4, 4, 10),
          'penalty': ['l1', 'l2'],
          'fit_intercept': [True, False]}

gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  cv=5,
                  scoring='accuracy',
                  return_train_score=True)

gs.fit(X_train, y_train)

# extract the grid search results

print('Best Parameters:')
print(gs.best_params_)
print('Best estimator C:')
print(gs.best_estimator_.C)
print('Best estimator mean cross validated training score:')
print(gs.best_score_)
print('Best estimator score on the full training set:')
print(gs.score(X_train, y_train))
print('Best estimator score on the test set:')
print(gs.score(X_test, y_test))
print('Best estimator coefficients:')
print(gs.best_estimator_.coef_)


# In[ ]:





# In[ ]:


# Can I find imp features (rev engg PCAs to real feature) - https://towardsdatascience.com/feature-extraction-using-principal-component-analysis-a-simplified-visual-demo-e5592ced100a


# In[ ]:


# Can i do deeper analytics to find which diseases are mostly like for readmssion or mots/leat expenseive so that we 

# can target the low hanging fruits


# In[ ]:


# Can i look at probablities and figure out the index and relate to the patient


# In[ ]:





# In[ ]:





# In[ ]:


final2008_2009DF_selectedFeatures['READMIT7'].unique()


# In[ ]:


final2008_2009DF_selectedFeatures['READMIT30'].unique()


# In[ ]:


final2008_2009DF_selectedFeatures['READMIT60'].unique()


# In[ ]:


final2008_2009DF_selectedFeatures['READMIT90'].unique()


# In[ ]:


# do oversample for eda

# Class count
count_class_0, count_class_1 = final2008_2009DF_selectedFeatures['DEAD'].value_counts() #READMIT7_FLAG
count_class_0, count_class_1


# In[ ]:


# Divide by class
df_class_0 = final2008_2009DF_selectedFeatures[final2008_2009DF_selectedFeatures['DEAD'] == 0]   
df_class_1 = final2008_2009DF_selectedFeatures[final2008_2009DF_selectedFeatures['DEAD'] == 1]
print(df_class_0.shape)
print(df_class_1.shape)


# In[ ]:


# Oversampling class 1

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
final2008_2009DF_over = pd.concat([df_class_0, df_class_1_over], axis=0)
print(final2008_2009DF_over.shape)
final2008_2009DF_over.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




