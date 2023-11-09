#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[4]:


housing= pd.read_csv("Housing.csv")


# In[5]:


housing.head()


# In[6]:


# number of observations
len(housing.index)


# In[7]:


housing.columns


# In[8]:


housing.isnull().sum()


# In[9]:


#filter only the number columns
df=housing.loc[:, [ 'price', 'area','bedrooms', 'bathrooms', 'stories', 'parking']]


# In[10]:


# recaling the variables (both)
#df_columns = df.columns
scaler = MinMaxScaler()
housing[['price']] = scaler.fit_transform(housing[['price']])
housing[['area']] = scaler.fit_transform(housing[['area']])
housing[['bedrooms']] = scaler.fit_transform(housing[['bedrooms']])
housing[['stories']] = scaler.fit_transform(housing[['stories']])
housing[['parking']] = scaler.fit_transform(housing[['parking']])


# In[11]:


# rename columns (since now its an np array)
#df = pd.DataFrame(df)
#df.columns = df_columns

housing.head()


# In[12]:


# list of all the "yes-no" binary categorical variables
# we'll map yes to 1 and no to 0
binary_vars_list =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']


# In[13]:


#, 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']]
#print(type(y))
#print(y)

from sklearn.preprocessing import LabelEncoder
label_encode=LabelEncoder()
le = LabelEncoder()
housing['mainroad']=le.fit_transform(housing['mainroad'])
housing['guestroom']=le.fit_transform(housing['guestroom'])
housing['basement']=le.fit_transform(housing['basement'])
housing['hotwaterheating']=le.fit_transform(housing['hotwaterheating'])
housing['airconditioning']=le.fit_transform(housing['airconditioning'])
housing['prefarea']=le.fit_transform(housing['prefarea'])
housing['furnishingstatus']=le.fit_transform(housing['furnishingstatus'])
print(housing.head())


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


# split into train and test
X_train, X_test, y_train,y_test = train_test_split(housing.drop('price',axis=1), housing['price'],
                                     train_size = 0.7,
                                     test_size = 0.3,
                                     random_state = 10)
print(len(X_train))
print(len(X_test))


# In[16]:


lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)


# In[17]:


from sklearn.metrics import r2_score


# In[18]:


r2 = r2_score(y_test, predictions)
print('r2 score for perfect model is', r2*100)


# In[ ]:




