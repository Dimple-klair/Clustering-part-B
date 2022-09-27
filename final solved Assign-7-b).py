#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# In[2]:


df=pd.read_csv('C:/Users/RIG1/Desktop/DS ASSIGNMENTS/QUESTIONS -all assignments/ASS 7/EastWestAirlines.csv')
df.head()


# In[3]:


df.info()


# #### standardization is required 
# #### As there is no col. with string values. so no need to select col. using ----------- iloc

# In[4]:


def norm_function(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# ##### as we can see 1st col ------ ID is not much required, so drop it

# In[5]:


#  ID# ----------------- is the column name
df1=df.drop(['ID#'],axis=1)


# In[6]:


df1


# # Now, create dendrogram--------------- using all the linkages

# In[10]:


plt.figure(figsize=(8, 7)) 
dendro=sch.dendrogram(sch.linkage(df1,method='complete'))


# In[11]:


plt.figure(figsize=(8, 7)) 
dendro=sch.dendrogram(sch.linkage(df1,method='average'))


# In[12]:


plt.figure(figsize=(8, 7)) 
dendro=sch.dendrogram(sch.linkage(df1,method='centroid'))


# In[43]:


plt.figure(figsize=(8, 7)) 
dendro=sch.dendrogram(sch.linkage(df1,method='single'))


# # Now, create cluster

# In[32]:


clust=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')


# In[33]:


clust


# In[34]:


# saving cluster for making charts


# In[35]:


y=clust.fit_predict(df1)


# In[36]:


cluster_df=pd.DataFrame(y,columns=['all_clusters'])


# In[37]:


cluster_df


# In[40]:


cluster_df['all_clusters'].value_counts()


# In[45]:


df1['all_clusters']=clust.labels_
df1


# In[42]:


# Plotting Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(df1['all_clusters'],df1['Balance'], c=clust.labels_) 


# In[47]:


df1.groupby('all_clusters').agg(['mean']).reset_index()


# In[ ]:




