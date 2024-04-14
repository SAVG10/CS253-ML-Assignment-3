#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#The main idea of this code is to use the random forest model.
#We will first go through all the columns of the data.
#We will be encoding string datatypes into numerical ones in order to easily deal with them.
#We will be renaming any complex names present in the dataset


# In[3]:


#Loads the data into the notebook
train_dat = pd.read_csv("/kaggle/input/who-is-the-real-winner/train.csv")
train_dat.columns


# In[4]:


#We will rename the constituency column
# Null values
train_dat.isnull().sum()


# In[5]:


#We will be goinh through all the individual columns for the datasets, and their datatypes.
#Will encode them as and when required.


# In[6]:


#ID
train_dat['ID'].unique()


# In[7]:


train_dat['ID'].dtype


# In[8]:


#Candidates
train_dat['Candidate'].unique()


# In[9]:


train_dat['Candidate'].dtype


# In[10]:


#Constituencies
train_dat = train_dat.rename(columns={'Constituency ∇': 'Constituency'})


# In[11]:


train_dat['Constituency'].unique()


# In[12]:


train_dat['Constituency'].dtype


# In[13]:


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


# In[14]:


#Parties
train_dat['Party'].unique()


# In[15]:


train_dat['Party'].dtype


# In[16]:


# Extracting the 'Party' column from the train_dat dataset
data = train_dat['Party']

# Using the label_encoder to transform the 'Party' column values into numerical labels, with adjustments
train_dat['Party'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Party' column
print(train_dat['Party'].unique())


# In[17]:


#Criminal Cases
train_dat['Criminal Case'].unique()


# In[18]:


train_dat['Criminal Case'].dtype


# In[19]:


#Total assets
train_dat['Total Assets'].unique()


# In[20]:


train_dat['Total Assets'].dtype


# In[21]:


#This means that we need to convert the datatypes of the Total Assets
#This can also imply that the content of Liabilites is also similar to the above and 
#Will be dealt similarly.


# In[22]:


#The liabilities in the data
train_dat['Liabilities'].unique()


# In[23]:


train_dat['Liabilities'].dtype


# In[24]:


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


# In[25]:


dat_1 = train_dat['Total Assets']
train_dat['Total Assets'] = [numero_uno_conv(value) for value in dat_1]
dat_2 = train_dat['Liabilities']
train_dat['Liabilities'] = [numero_uno_conv(value) for value in dat_2]


# In[26]:


train_dat['Total Assets'].dtype


# In[27]:


train_dat['Liabilities'].dtype


# In[28]:


#We can see the presence of nill values in the dataset
train_dat.isnull().sum()


# In[29]:


#States
train_dat['state'].unique()


# In[30]:


train_dat['state'].dtype


# In[31]:


# Extracting the 'state' column from the train_dat dataset
data = train_dat['state']

# Initializing a LabelEncoder instance
label_encoder = LabelEncoder()

# Encoding the 'state' column values using the label_encoder, with adjustments
train_dat['state'] = label_encoder.fit_transform(data) + 1

# Displaying the unique encoded values of the 'state' column
print(train_dat['state'].unique())


# In[32]:


#Education
train_dat['Education'].unique()


# In[33]:


train_dat['Education'].dtype


# In[34]:


#We will be encoding Total Assets

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = train_dat['Total Assets']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
train_dat['Total Assets'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(train_dat['Total Assets'].unique())


# In[35]:


#Same for liabilities

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = train_dat['Liabilities']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
train_dat['Liabilities'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(train_dat['Liabilities'].unique())


# In[36]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


# In[37]:


#Now, we have read the training data, now is the time we repeat the steps with the test data.


# In[38]:


test_dat = pd.read_csv("/kaggle/input/who-is-the-real-winner/test.csv")


# In[39]:


test_dat.columns


# In[40]:


#The objects are similar to that of the training dataset, so lets do the same with the test dataset.
#We won't be testing the dataset and inspecting the data, but simply convert to the required type and 
#Encode the data as and when required.


# In[41]:


#Constituency
test_dat = test_dat.rename(columns={'Constituency ∇': 'Constituency'})


# In[42]:


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


# In[43]:


# Extracting the 'Party' column from the train_dat dataset
data = test_dat['Party']

# Using the label_encoder to transform the 'Party' column values into numerical labels, with adjustments
test_dat['Party'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Party' column
print(test_dat['Party'].unique())


# In[44]:


#Conversion of the Assets and Liabilities into the numerical values
data1 = test_dat['Total Assets']
test_dat['Total Assets'] = [numero_uno_conv(value) for value in data1]
data2 = test_dat['Liabilities']
test_dat['Liabilities'] = [numero_uno_conv(value) for value in data2]


# In[45]:


# Extracting the 'state' column from the train_dat dataset
data = test_dat['state']

# Initializing a LabelEncoder instance
label_encoder = LabelEncoder()

# Encoding the 'state' column values using the label_encoder, with adjustments
test_dat['state'] = label_encoder.fit_transform(data) + 1

# Displaying the unique encoded values of the 'state' column
print(test_dat['state'].unique())


# In[46]:


#Ranging to the total assets first and verify it.
#test_dat['Total Assets'] = [assign_range(val) for val in test_dat['Total Assets']]
test_dat['Total Assets'].unique()


# In[47]:


#Ranging to the total assets first and verify it.
#test_dat['Liabilities'] = [assign_range(val) for val in test_dat['Liabilities']]
test_dat['Liabilities'].unique()


# In[48]:


#We will be encoding Total Assets

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = test_dat['Total Assets']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
test_dat['Total Assets'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(test_dat['Total Assets'].unique())


# In[49]:


#Same for liabilities

# Extracting the 'Total Assets' column from the train_dat DataFrame
data = test_dat['Liabilities']

# Initializing a LabelEncoder object
label_encoder = LabelEncoder()

# Encoding the 'Total Assets' column values using the label_encoder, with adjustments
test_dat['Liabilities'] = label_encoder.fit_transform(data) + 1

# Displaying the unique numerical labels generated after encoding the 'Total Assets' column
print(test_dat['Liabilities'].unique())


# In[50]:


#Now, we will start the plotting and prediction
test = test_dat.drop(['ID', 'Candidate', 'Constituency'], axis=1)


# In[51]:


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


# In[52]:


y_pred = rf_class.predict(test)


# In[53]:


result = pd.DataFrame(y_pred)


# In[54]:


result.columns = ['Education']


# In[55]:


submission = pd.DataFrame(test_dat['ID'])


# In[56]:


submission['Education'] =  result['Education']
submission.reset_index(drop=True , inplace=True)


# In[57]:


submission.to_csv('/kaggle/working/submission.csv', index= False)


# In[58]:


import matplotlib.pyplot as plt

# Plot histogram for Total Assets
plt.hist(train_dat['Total Assets'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Assets')
plt.xlabel('Total Assets')
plt.ylabel('Frequency')
plt.show()


# In[59]:


import matplotlib.pyplot as plt

# Plot histogram for Liabilities
plt.figure(figsize=(8, 6))
plt.hist(train_dat['Liabilities'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Liabilities')
plt.xlabel('Liabilities')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[60]:


import matplotlib.pyplot as plt

# Count the frequency of each party
party_counts = train_dat['Party'].value_counts()

# Plot bar plot for Party
plt.figure(figsize=(10, 6))
party_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Parties')
plt.xlabel('Party')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[61]:


import matplotlib.pyplot as plt

# Count the frequency of each value in 'Criminal Cases'
criminal_cases_counts = train_dat['Criminal Case'].value_counts()

# Plot bar plot for Criminal Cases
plt.figure(figsize=(10, 6))
criminal_cases_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Criminal Cases')
plt.xlabel('Criminal Cases')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[62]:


import matplotlib.pyplot as plt

# Count the frequency of each value in 'Education'
education_counts = train_dat['Education'].value_counts()

# Plot bar plot for Education
plt.figure(figsize=(10, 6))
education_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# In[63]:


import matplotlib.pyplot as plt

# Count the frequency of each value in 'state'
state_counts = train_dat['state'].value_counts()

# Plot bar plot for state
plt.figure(figsize=(10, 6))
state_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of States')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

