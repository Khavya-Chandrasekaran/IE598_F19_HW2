#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import necessary modules
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


df=pd.read_csv('K:\Fall 2019\MLF\Assignments\HW2\Treasury Squeeze test - DS1.csv', header=None)
# Create feature and target arrays
X = df.iloc[1:,2:11]
y = df.iloc[1:,11] 

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
dt1= DecisionTreeClassifier(criterion='gini', random_state=1)
# Fit dt_entropy to the training set
dt.fit(X_train, y_train)
dt1.fit(X_train,y_train)

# Print the accuracy
# Predict test set labels
y_pred = dt.predict(X_test)
y_pred1=dt1.predict(X_test)

# Compute test set accuracy  
accuracy_entropy = dt.score(X_test, y_test)
accuracy_gini = dt1.score(X_test, y_test)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)

print("My name is Khavya Chandrasekaran")
print("My NetID is: khavyac2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




