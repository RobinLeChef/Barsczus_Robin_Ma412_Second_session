# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:39:30 2024

@author: robin
"""

#%% Set up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#%% 
"""
            Part 1: Analysis of train and test data

"""

# In this part, run the code section by section to not subplot the graphs

#%% Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.head()
test_data.head()

train_data.drop('id', axis = 1)
test_data.drop('id', axis = 1)

#%% Number of customers based on satisfaction
s = sns.countplot(x='satisfaction',data=train_data)
abs_values = train_data['satisfaction'].value_counts().values

s.bar_label(container=s.containers[0], labels=abs_values);

#%% Count of customers nased on gender
sns.countplot(x='Gender',data=train_data)

#%% Customer satisfaction based on customer type
sns.countplot(x='satisfaction',data=train_data, hue='Customer Type', palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title= 'Customer Type', borderaxespad=0.)

#%% Customer satisfaction base based on type of travel
sns.countplot(x='satisfaction',data=train_data, hue='Type of Travel', palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title= 'Type of Travel', borderaxespad=0.)

#%% Customer satisfaction base based on class of travel
sns.countplot(x='satisfaction',data=train_data, hue='Class', palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title= 'Type of Travel', borderaxespad=0.)

#%% Check the distribution of customer age
sns.set_style('whitegrid')
sns.histplot(x='Age', kde= True, data=train_data,bins=30)

#%% Customer Profile based Age and Travel Type
sns.boxplot(x="Type of Travel", y="Age", data=train_data, palette="viridis")

#%% Flight distance compare to the Class
sns.barplot(x='Class',y='Flight Distance',data=train_data)

#%% Flight distance compare to Customer type
sns.barplot(x='Customer Type',y='Flight Distance',data=train_data)

#%%
plt.figure(figsize=(12,8), dpi =200)
sns.heatmap(train_data.corr(),cmap='coolwarm',annot=True)

#%% Check for missing values
train_data.isnull().sum()

#%% Fill missing data with mean value of column
train_data['Arrival Delay in Minutes'] = train_data['Arrival Delay in Minutes'].fillna(value = train_data['Arrival Delay in Minutes'].mean())

#Verify that missing values were replaced. 
train_data.isnull().sum()

# Check for duplicated data
train_data.duplicated().sum()

#%% 
"""
            Part 2: Machine learning model

"""
#%% Data Classification Analysis

# Print columns to verify the existence of 'satisfaction'
print(train_data.columns)
print(test_data.columns)
# If 'satisfaction' don't exist, reload data and fill missing data

from sklearn.impute import SimpleImputer

# For numerical columns
num_imputer = SimpleImputer(strategy='median')  
# For categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')  

# Apply the appropriate imputer to each subset of columns
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_data.select_dtypes(include=['object', 'category']).columns

train_data[numerical_cols] = num_imputer.fit_transform(train_data[numerical_cols])
train_data[categorical_cols] = cat_imputer.fit_transform(train_data[categorical_cols])

# Do the same for test_data
test_data[numerical_cols] = num_imputer.transform(test_data[numerical_cols])
test_data[categorical_cols] = cat_imputer.transform(test_data[categorical_cols])

# Encode categorical variables using get_dummies
categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
train_data = pd.get_dummies(train_data, columns=categorical_features)
test_data = pd.get_dummies(test_data, columns=categorical_features)

# Align columns of train and test sets
train_data, test_data = train_data.align(test_data, join='inner', axis=1)

# Drop the target variable from X data sets
X_train = train_data.drop('satisfaction', axis=1)
y_train = train_data['satisfaction']
X_test = test_data.drop('satisfaction', axis=1)
y_test = test_data['satisfaction']

# Scale the data
scaler = StandardScaler()
scld_X_train = scaler.fit_transform(X_train)
scld_X_test = scaler.transform(X_test)

# Set up instances for models being used
log_model = LogisticRegression()
knn_model = KNeighborsClassifier(n_neighbors=3)
svm_model = SVC()
rfc_model = RandomForestClassifier()
    
# Model initialization
models = {
    'Logistic Regression': log_model,
    'K-Nearest Neighbors': knn_model,
    'Support Vector Machine': svm_model,
    'Random Forest': rfc_model
}

# Model training and evaluation
for model_name, model in models.items():
    model.fit(scld_X_train, y_train)
    y_pred = model.predict(scld_X_test)
    print(f'Analysis with Model: {model_name}')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.grid(False)
    plt.show()
    print('***********************************************************************************')

#%% 
"""
            Part 3: Prediction for customer satisfaction

"""

#%% Coefficients based on logistic regression model
coefficients = pd.DataFrame(index=X_test.columns,data=log_model.coef_ .reshape(-1,1) ,columns=['Coefficient'])
plt.figure(figsize=(14,8),dpi=200)
sns.barplot(data=coefficients.sort_values('Coefficient'),x=coefficients.sort_values('Coefficient').index,y='Coefficient')
plt.title('Variable Coefficients Based on The Logistic Regression  Model')
plt.xticks(rotation=90);

#%% Coefficients based on Random Forest Model
var_importance = pd.DataFrame(index=X_test.columns,data=rfc_model.feature_importances_ .reshape(-1,1) ,columns=['Importance'])
plt.figure(figsize=(12,8),dpi=200)
sns.barplot(data=var_importance.sort_values('Importance'),x=var_importance.sort_values('Importance').index,y='Importance')
plt.title('Variable Importance Based on The Random Forest Model')
plt.xticks(rotation=90);
