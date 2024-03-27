#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder


# In[2]:


df=pd.read_csv("Housing.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.info()


# In[8]:


df.shape


# In[9]:


df.shape


# In[10]:


df.describe()


# In[11]:


df.isnull()


# In[12]:


df.isnull().sum()


# In[13]:


df.columns


# In[14]:


sns.pairplot(df)


# In[26]:


df.head()


# In[15]:


sns.jointplot(x='area',y='price',data=df)


# In[16]:


sns.jointplot(x='bedrooms',y='price',data=df)


# In[17]:


sns.lmplot(x='area',y='price',data=df)


# In[19]:


df.corr


# In[32]:


main_road = LabelEncoder()
df['mainroad'] = main_road.fit_transform(df['mainroad'])

guest_room = LabelEncoder()
df['guestroom'] = guest_room.fit_transform(df['guestroom'])

base_ment = LabelEncoder()
df['basement'] = base_ment.fit_transform(df['basement'])

hotwater_heating = LabelEncoder()
df['hotwaterheating'] = hotwater_heating.fit_transform(df['hotwaterheating'])

air_conditioning = LabelEncoder()
df['airconditioning'] = air_conditioning.fit_transform(df['airconditioning'])

furnishing_status = LabelEncoder()
df['furnishingstatus'] = furnishing_status.fit_transform(df['furnishingstatus'])

pref_area = LabelEncoder()
df['prefarea'] = pref_area.fit_transform(df['prefarea'])


# In[33]:


df.head()


# In[34]:


#sns.heatmap(df.corr() ,annot=True )
sns.heatmap(df.corr(), annot = True, linewidths = .5, cmap = plt.cm.cool)


# In[36]:


lst =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

def to_values(x):
    return x.map({'yes': 1, 'no': 0})

df[lst] = df[lst].apply(to_values)


# In[37]:


df.head()


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X=df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
X.head()


# In[40]:


y=df[['price']]
y.head()


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[42]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[43]:


lm.fit(X,y)


# In[44]:


lm.coef_


# In[45]:


lm.intercept_


# In[46]:


predi=lm.predict(X_test)


# In[47]:


predi


# In[48]:


predi


# In[49]:


plt.scatter(x=y_test ,y=predi)
plt.xlabel('Y')
plt.ylabel('Predicted ')


# In[50]:


from sklearn import metrics


# In[51]:


print('MAE:', metrics.mean_absolute_error(y_test, predi))
print('MSE:', metrics.mean_squared_error(y_test, predi))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predi)))


# In[ ]:




