#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = pd.read_csv(r"C:\Users\munee\OneDrive\Documents\Desktop\data set\air_quality_health_impact_data.csv")


# In[3]:


data.info()


# In[4]:


data.index


# In[5]:


data.columns


# In[6]:


data.describe


# In[7]:


data.head()


# In[8]:


data.drop_duplicates()
data


# In[9]:


data.dropna()


# In[10]:


data.RecordID


# In[11]:


print(data.isna().sum())


# In[12]:


# Correlation matrix to see relationships between variables
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Air Quality and Health Impacts')
plt.show()


# In[18]:


print(data.columns)


# In[22]:


print(data.columns)


# In[26]:


if 'date' in data.columns and 'air_quality_metric' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')


# In[30]:


print(data.columns)


# In[38]:


import seaborn as sns
import matplotlib as plt


# In[39]:


subdata = data[['AQI', 'PM10','PM2_5','NO2','SO2','O3']]
sns.pairplot(subdata)
    


# In[40]:


corr_matrix = data.corr()


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[46]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='AQI', y='HealthImpactScore', data=data)
plt.title('AQI vs Health Impact Score')
plt.show()


# In[47]:


cases_df = data[['RespiratoryCases', 'CardiovascularCases']].sum()

plt.figure(figsize=(8,6))
cases_df.plot(kind='bar', color=['blue', 'orange'])
plt.title('Comparison of Respiratory and Cardiovascular Cases')
plt.ylabel('Total Cases')
plt.show()


# In[48]:


plt.figure(figsize=(12,6))


# In[49]:


# AQI Distribution
plt.subplot(1, 2, 1)
sns.histplot(data['AQI'], kde=True, color='blue')
plt.title('Distribution of AQI')


# In[51]:


# HealthImpactScore Distribution
plt.subplot(1, 2, 2)
sns.histplot(data['HealthImpactScore'], kde=True, color='green')
plt.title('Distribution of Health Impact Score')
plt.tight_layout()
plt.show()


# In[ ]:




