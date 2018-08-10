
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


# In[68]:


data = np.array(pd.read_csv('data.csv'))
X = data[:,:-1]
y = data[:,-1]


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[70]:


regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)


# In[71]:


plt.scatter(X_test[:,1],y_test,label='Original Data')
plt.scatter(X_test[:,1],y_pred,label='Predicted Data')
plt.legend()
plt.xlabel('Dose')
plt.ylabel('Output')
plt.savefig('Plot 1.png')
plt.clf()

# In[72]:


plt.scatter(X_test[:,2],y_test,label='Original Data')
plt.scatter(X_test[:,2],y_pred,label='Predicted Data')
plt.legend()
plt.xlabel('Energy')
plt.ylabel('Output')
plt.savefig('Plot 2.png')
plt.clf()

# In[73]:


plt.scatter(X_test[:,3],y_test,label='Original Data')
plt.scatter(X_test[:,3],y_pred,label='Predicted Data')
plt.legend()
plt.xlabel('Angle')
plt.ylabel('Output')
plt.savefig('Plot 3.png')


# In[74]:


pickle.dump(regr,open('Linear Regression Model','wb'))


# In[75]:


print('r2 Score for model is',r2_score(y_test,y_pred))


# In[63]:


print('Mean squared error is',mean_squared_error(y_test,y_pred))

