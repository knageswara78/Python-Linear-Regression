#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction ( Linear Regression)

# ## Shortcuts

# ### Run the current cell and select below
# Shift + Enter 
# ### Save and checkpoint
# Ctrl + S
# ### Create a cell above

# In[ ]:




# Esc,Shift,a
# ### Delete cell

# In[ ]:


# Esc, dd


# 
# 

# ### Run all freshly

# In[ ]:


# Kernel -> Restart & Run All


# ## Import Libraries

# In[1]:


# Importing libraries
import pandas


# ## Read File

# In[2]:


# Read a file
pandas.read_csv('USA_Housing.csv')


# ## Store File after reading

# In[3]:


pandas.read_csv('USA_Housing.csv')


# In[4]:


df = pandas.read_csv('USA_Housing.csv')


# ## Display first 5 rows

# In[5]:


df.head(5)


# In[6]:


df.tail()


# ## Get no.of rows and columns

# In[7]:


df.shape


# ## See data type of each variable

# In[8]:


df.info


# In[9]:


df.info()


# ## See base statistics of each variable

# In[10]:


df.describe # without brackets format is not in proper way


# In[11]:


df.describe()


# ## Get all the variable names
# 

# In[12]:


# df.columns()


# In[13]:


df.columns


# ## Training a Model

# ### Separate Input and Output Variables ( X and y arrays )

# In[29]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']] # 'Address'


# In[30]:


y = df['Price']


# ## Train Test Split

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X.shape


# In[33]:


y.shape


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101) # train : 80% and test: 20%


# In[35]:


X_train.shape


# In[36]:


y_train.shape


# In[37]:


X_test.shape


# In[38]:


y_test.shape


# In[ ]:





# ## Creating and Training the Model

# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


lr = LinearRegression()


# In[41]:


lr.fit(X_train,y_train)


# ## Model Evaluation

# In[42]:


# print the intercept
print(lr.intercept_) # B0


# In[43]:


coeff_df = pandas.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
coeff_df


# ## Make Predictions on test data

# In[44]:


predictions = lr.predict(X_test)


# In[45]:


predictions


# ## Find Regression Evaluation Metrics 
# ### (Mean Absolute Error,  Mean Squared Error and  Root Mean Squared Error)

# In[46]:


from sklearn import metrics


# In[47]:


import numpy


# In[48]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', numpy.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:





# In[ ]:




