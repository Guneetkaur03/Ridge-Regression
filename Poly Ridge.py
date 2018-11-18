
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error,r2_score


# In[2]:


df = pd.read_csv('C:/Users/Student/Desktop/ml self/kunal/kunal.csv')

X = df[['Pulse-on Time','Servo voltage','wire feed','wire tension']].astype(float) # here we have 4 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
y = df['ER'].astype(float)


# In[3]:


X_train = X[:-6]
X_test = X[-6:]
y_train = y[:-6]
y_test = y[-6:]


# In[4]:


poly = PolynomialFeatures(degree=3,interaction_only=False)
X_train_new = poly.fit_transform(X_train)
X_test_new = poly.fit_transform(X_test)

scaler  = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train_new)
X_test_scale = scaler.transform(X_test_new)



# In[5]:


alphas = 10**np.linspace(10,-2,100)*0.5
alphas


# In[6]:


ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)


# In[7]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')


# In[12]:



ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_
ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge4.fit(X_train_scale, y_train)
print(mean_squared_error(y_test, ridge4.predict(X_test_scale)))

#ridge = RidgeCV(alphas = 0.9, normalize = True)
#model = ridge.fit(X_train_scale, y_train)             # Fit a ridge regression on the training data
#pred = ridgecv.predict(X_test_scale)           # Use this model to predict the test data
#print(ridge4.coef_) # Print coefficients
#print(mean_squared_error(y_test, pred))     
y_pred_er = ridge4.predict(X_train_scale)
print(r2_score(y_train, y_pred_er))


# In[10]:


print("Predicted values of KW for test data are :")
pred = ridge4.predict(X_test_scale)
print(ridge4.predict(X_test_scale))
print("Actual values of KW for test data are :")
print(y_test)


# In[11]:


test = [302.689, 292.041, 271.998, 325.289, 331.386, 311.654]
sum = 0
for i in range(len(test)):
    cal = abs(test[i]-pred[i])
    j = cal/test[i]
    sum = sum + j
print(sum/6)
    

