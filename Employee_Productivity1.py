#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv('garments_worker_productivity.csv')


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.head(8)


# In[9]:


df.info


# In[11]:


df.describe


# In[13]:


## Null Values


# In[14]:


df.isnull().sum()


# In[29]:


df=df.dropna(how='all')


# In[30]:


df


# In[31]:


df.isnull().sum()


# In[32]:


df=df.interpolate()


# In[35]:


df


# In[33]:


df.isnull().sum()


# In[34]:


print(df.duplicated())


# In[36]:


df.drop_duplicates(inplace = True)


# In[37]:


df


# In[38]:


df.describe()


# In[39]:


prod_filter=df['actual_productivity']>1 


# In[40]:


prod_filter.sum()


# In[41]:


prod1_filter=df['actual_productivity']<=1


# In[42]:


df=df.loc[prod1_filter, :]


# In[43]:


df


# In[45]:


df['department'].unique() 


# In[46]:


df


# In[47]:


df['quarter'].unique()


# In[49]:


dep_dic = {'sewing': 1, 'finishing': 2}
df1['department'] = df['department'].map(dep_dic)


# In[51]:


df


# In[52]:


Q_dic = {'Quarter1':1, 'Quarter2':2, 'Quarter3':3, 'Quarter4':4, 'Quarter5':5}


# In[55]:


df1['quarter'] = df['quarter'].map(Q_dic)


# In[56]:


df


# In[85]:


df['day'].unique()


# In[90]:


day_dic = {'Thursday': 1, 'Saturday': 2, 'Sunday': 3, 'Monday': 4, 'Tuesday': 5, 'Wednesday': 6}


# In[94]:


df['day'] = df['day'].map(day_dic)


# In[95]:


df


# In[ ]:





# In[96]:


df.dtypes


# In[ ]:


df = df.drop('date', axis=1) 


# In[98]:


df.head()


# In[99]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[101]:


x,y = plt.subplots(figsize=(12,9))
sns.heatmap(df.corr(), cmap='YlGnBu', square=True, linewidth=.5, annot=True)
plt.show()


# In[102]:


df['department'].value_counts().plot.pie()


# In[103]:


df['actual_productivity'].value_counts()


# In[110]:


df['actual_productivity'].value_counts()


# In[106]:


df.shape


# In[113]:


df.actual_productivity.min()


# In[114]:


df.actual_productivity.max()


# In[115]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# In[116]:


corr_matrix = df.corr() #feature selection


# In[117]:


corr_matrix['no_of_workers'].sort_values(ascending = False)


# In[126]:


df = df.drop('department', axis=1)  #least feature
df.head()


# In[127]:


df.shape


# In[128]:


#checking or correcting imbalances - outliers


# In[129]:


q1 = df.quantile(0.25)


# In[130]:


q3 = df.quantile(0.75)


# In[132]:


IQR =q3 -q1  #inter quartile range


# In[133]:


print(IQR)


# In[134]:


df = df[~((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5*IQR))).any(axis=1)]  # lower and upper bound outlier


# In[135]:


print(df.shape)


# In[136]:


df.columns


# In[137]:


# WEEK 1 finished


# In[ ]:




