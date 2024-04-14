#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[4]:


#The main idea of this code is to use the random forest model.
#We will first go through all the columns of the data.
#We will be encoding string datatypes into numerical ones in order to easily deal with them.
#We will be renaming any complex names present in the dataset


# In[5]:


#Loads the data into the notebook
train_dat = pd.read_csv("/kaggle/input/who-is-the-real-winner/train.csv")
train_dat.columns


# In[6]:


#We will rename the constituency column
# Null values
train_dat.isnull().sum()


# In[7]:


#We will be goinh through all the individual columns for the datasets, and their datatypes.
#Will encode them as and when required.


# In[8]:


#ID
train_dat['ID'].unique()


# In[9]:


train_dat['ID'].dtype


# In[10]:


#Candidates
train_dat['Candidate'].unique()


# In[11]:


train_dat['Candidate'].dtype


# In[12]:


#Constituencies
train_dat = train_dat.rename(columns={'Constituency ∇': 'Constituency'})


# In[13]:


train_dat['Constituency'].unique()


# In[14]:


train_dat['Constituency'].dtype


# In[15]:


# Importing the LabelEncoder module from the scikit-learn library
from sklearn.preprocessing import LabelEncoder

# Extract the 'Constituency' column from the train_dat dataframe
data = train_dat['Constituency']

# Initialize a label encoder object
label_encoder = LabelEncoder()

# Fit the label encoder to the data and transform the categories into integers
encoded_data = label_encoder.fit_transform(data)

# Add 1 to each encoded value to ensure no encoded value is 0
encoded_data += 1

# Replace the original 'Constituency' column in train_dat with the encoded values
train_dat['Constituency'] = encoded_data

# Print the unique encoded values of the 'Constituency' column
print(train_dat['Constituency'].unique())


# In[16]:


#Parties
train_dat['Party'].unique()


# In[17]:


train_dat['Party'].dtype


# In[18]:


# Extracting the 'Party' column from the train_dat dataset
data = train_dat['Party']

# Using the label_encoder to transform the 'Party' column values into numerical labels, with adjustments
train_dat['Party'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Party' column
print(train_dat['Party'].unique())


# In[19]:


#Criminal Cases
train_dat['Criminal Case'].unique()


# In[20]:


train_dat['Criminal Case'].dtype


# In[21]:


#Total assets
train_dat['Total Assets'].unique()


# In[22]:


train_dat['Total Assets'].dtype


# In[23]:


#This means that we need to convert the datatypes of the Total Assets
#This can also imply that the content of Liabilites is also similar to the above and 
#Will be dealt similarly.


# In[24]:


#The liabilities in the data
train_dat['Liabilities'].unique()


# In[25]:


train_dat['Liabilities'].dtype


# In[26]:


#Data conversion for Total Assets and Liabilites
def numero_uno_conv(value):
    value = str(value)
    if value.endswith('Crore+'):
        return float(value.replace(' Crore+', '')) * 10000000
    elif value.endswith('Crore'):
        return float(value.replace(' Crore', '')) * 10000000
    elif value.endswith('Lac+'):
        return float(value.replace(' Lac+', '')) * 100000
    elif value.endswith('Lac'):
        return float(value.replace(' Lac', '')) * 100000
    elif value.endswith('Thou+'):
        return float(value.replace(' Thou+', '')) * 1000
    elif value.endswith('Thou'):
        return float(value.replace(' Thou', '')) * 1000
    elif value.endswith('Hund+'):
        return float(value.replace(' Hund+', '')) * 100
    elif value.endswith('Hund'):
        return float(value.replace(' Hund', '')) * 100
    else:
        return float(value.replace('+', ''))


# In[27]:


dat_1 = train_dat['Total Assets']
train_dat['Total Assets'] = [numero_uno_conv(value) for value in dat_1]
dat_2 = train_dat['Liabilities']
train_dat['Liabilities'] = [numero_uno_conv(value) for value in dat_2]


# In[28]:


train_dat['Total Assets'].dtype


# In[29]:


train_dat['Liabilities'].dtype


# In[30]:


#We can see the presence of nill values in the dataset
train_dat.isnull().sum()


# In[31]:


#States
train_dat['state'].unique()


# In[32]:


train_dat['state'].dtype


# In[33]:


# Extracting the 'state' column from the train_dat dataset
data = train_dat['state']

# Initializing a LabelEncoder instance
label_encoder = LabelEncoder()

# Encoding the 'state' column values using the label_encoder, with adjustments
train_dat['state'] = label_encoder.fit_transform(data) + 1

# Displaying the unique encoded values of the 'state' column
print(train_dat['state'].unique())


# In[34]:


#Education
train_dat['Education'].unique()


# In[35]:


train_dat['Education'].dtype


# In[36]:


#We will be encoding Total Assets

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = train_dat['Total Assets']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
train_dat['Total Assets'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(train_dat['Total Assets'].unique())


# In[37]:


#Same for liabilities

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = train_dat['Liabilities']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
train_dat['Liabilities'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(train_dat['Liabilities'].unique())


# In[38]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


# In[39]:


#Now, we have read the training data, now is the time we repeat the steps with the test data.


# In[40]:


test_dat = pd.read_csv("/kaggle/input/who-is-the-real-winner/test.csv")


# In[41]:


test_dat.columns


# In[42]:


#The objects are similar to that of the training dataset, so lets do the same with the test dataset.
#We won't be testing the dataset and inspecting the data, but simply convert to the required type and 
#Encode the data as and when required.


# In[43]:


#Constituency
test_dat = test_dat.rename(columns={'Constituency ∇': 'Constituency'})


# In[44]:


# Extract the 'Constituency' column from the train_dat dataframe
data = test_dat['Constituency']

# Initialize a label encoder object
label_encoder = LabelEncoder()

# Fit the label encoder to the data and transform the categories into integers
encoded_data = label_encoder.fit_transform(data)

# Add 1 to each encoded value to ensure no encoded value is 0
encoded_data += 1

# Replace the original 'Constituency' column in train_dat with the encoded values
test_dat['Constituency'] = encoded_data

# Print the unique encoded values of the 'Constituency' column
print(test_dat['Constituency'].unique())


# In[45]:


# Extracting the 'Party' column from the train_dat dataset
data = test_dat['Party']

# Using the label_encoder to transform the 'Party' column values into numerical labels, with adjustments
test_dat['Party'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Party' column
print(test_dat['Party'].unique())


# In[46]:


#Conversion of the Assets and Liabilities into the numerical values
data1 = test_dat['Total Assets']
test_dat['Total Assets'] = [numero_uno_conv(value) for value in data1]
data2 = test_dat['Liabilities']
test_dat['Liabilities'] = [numero_uno_conv(value) for value in data2]


# In[47]:


# Extracting the 'state' column from the train_dat dataset
data = test_dat['state']

# Initializing a LabelEncoder instance
label_encoder = LabelEncoder()

# Encoding the 'state' column values using the label_encoder, with adjustments
test_dat['state'] = label_encoder.fit_transform(data) + 1

# Displaying the unique encoded values of the 'state' column
print(test_dat['state'].unique())


# In[48]:


#Ranging to the total assets first and verify it.
#test_dat['Total Assets'] = [assign_range(val) for val in test_dat['Total Assets']]
test_dat['Total Assets'].unique()


# In[49]:


#Ranging to the total assets first and verify it.
#test_dat['Liabilities'] = [assign_range(val) for val in test_dat['Liabilities']]
test_dat['Liabilities'].unique()


# In[50]:


#We will be encoding Total Assets

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = test_dat['Total Assets']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
test_dat['Total Assets'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(test_dat['Total Assets'].unique())


# In[51]:


#Same for liabilities

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = test_dat['Liabilities']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
test_dat['Liabilities'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(test_dat['Liabilities'].unique())


# In[52]:


#Now, we will start the plotting and prediction
test = test_dat.drop(['ID', 'Candidate', 'Constituency'], axis=1)


# In[54]:


from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split


X = train_dat.drop(['Education','ID', 'Candidate', 'Constituency'], axis=1)  
y = train_dat['Education']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

rf_class = RandomForestClassifier(n_estimators=300, min_samples_leaf = 2, min_samples_split = 5, random_state=100)  

rf_class.fit(X, y)

y_pred = rf_class.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 score on test set:", f1)


# In[56]:


y_pred = rf_class.predict(test)


# In[57]:


result = pd.DataFrame(y_pred)


# In[58]:


result.columns = ['Education']


# In[59]:


submission = pd.DataFrame(test_dat['ID'])


# In[61]:


submission['Education'] =  result['Education']
submission.reset_index(drop=True , inplace=True)


# In[62]:


submission.to_csv('/kaggle/working/submission.csv', index= False)

