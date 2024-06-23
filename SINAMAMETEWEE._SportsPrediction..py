#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#from google.colab import drive

# Mounting the drive to access FIFA 23 Player Dataset
#drive.mount('/content/drive')


# In[2]:


# Loading the training dataset
import pandas as pd
train_file_path = "C:\\Users\\Administrator\\male_players (legacy).csv"
df = pd.read_csv(train_file_path)
# Dropping columns with 30% or more null values
df.dropna(thresh= 0.3 * len(df), axis=1, inplace=True)

df.head()


# In[3]:


# Handling numeric data
numeric_data = df.select_dtypes(include='number')
columns_with_nans = numeric_data.columns[numeric_data.isna().any()].tolist()

# Remove columns with null values
numeric_data = numeric_data.drop(columns=columns_with_nans)

# Filling all missing values using the median
imputer = SimpleImputer(strategy="median")
df_imputed = imputer.fit_transform(df[columns_with_nans])
df_imputed = pd.DataFrame(df_imputed, columns=columns_with_nans)

# Concatenate the imputed data back to the numeric dataset
numeric_data = pd.concat([numeric_data, df_imputed], axis=1)

# Confirm no null values
numeric_data.isnull().sum()


# In[4]:


# Handling categorical data
alphabet_data = df.select_dtypes(exclude='number')
columns_with_nans = alphabet_data.columns[alphabet_data.isna().any()].tolist()

# Dropping columns with null values
alphabet_data = alphabet_data.drop(columns_with_nans, axis=1)

# Imputing missing values in categorical data
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = alphabet_data.copy()
df_imputed[columns_with_nans] = imputer.fit_transform(df[columns_with_nans])
df_imputed = pd.DataFrame(df_imputed, columns=columns_with_nans)

# Concatenate the imputed data back to the categorical dataset
alphabet_data = pd.concat([alphabet_data, df_imputed], axis=1)

# Confirm no null values
alphabet_data.isnull().sum()


# In[5]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# a copy of the DataFrame to avoid modifying the original data
encoded_df = alphabet_data.copy()

for col in alphabet_data.columns:
    # Convert the column to string type before encoding
    encoded_df[col] = label_encoder.fit_transform(alphabet_data[col].astype(str))

alphabet_data=encoded_df


# In[6]:


# a dataframe of the cleaned dataset which contains usable string components and numeric components from the dataset
new_DataSet=pd.concat([alphabet_data,numeric_data],axis=1)
new_DataSet.head()


# In[ ]:


# Calculate the correlation matrix for overall
correlation_matrix = new_DataSet.corr()['overall']

high_correlations = correlation_matrix[(correlation_matrix > 0.48) | (correlation_matrix < -0.48)]
high_correlations


# In[ ]:


df[high_correlations.index]


# In[ ]:


#ploting the data on linear plot
x = new_DataSet['potential']
y = new_DataSet['overall']

# Plotting
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data')

plt.xlabel('Player Capability')
plt.ylabel('Overall Performance')
plt.title('Line Plot showing Player Potential against Overall Performance')
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:


# Scaling the columns for better prediciton
sc = StandardScaler()

# Finding scaler of relevantDataSet values without 'overall' which is the dependent variable
scaled = sc.fit_transform(new_DataSet[high_correlations.index].loc[:,"potential":])
scaled


# In[ ]:


# now on to working with the test data
test_data = pd.read_csv("C:\\Users\\Administrator\\players_22-1.csv")
new_DataSet2 = test_data[high_correlations.index]

# Dropping columns with null values
new_DataSet2.dropna(thresh= 0.3 * len(test_data), axis=1, inplace=True)

numeric_data = new_DataSet2.select_dtypes(include='number')
columns_with_nans = numeric_data.columns[numeric_data.isna().any()].tolist()

# Dropping columns with null values (Nans)
numeric_data = numeric_data.drop(columns=columns_with_nans)
imputer = SimpleImputer(strategy="median")

# Imputing the data
data_imputed = imputer.fit_transform(new_DataSet2[columns_with_nans])
data_imputed = pd.DataFrame(df_imputed, columns=columns_with_nans)

# Concating the imputed dataframe with the numeric dataframe
numeric_data = pd.concat([numeric_data,data_imputed],axis=1)

# Since all our data is numerical
new_DataSet2 = numeric_data

# Scaling the columns to make better to prediciton
sc = StandardScaler()

# scaling it
scaled = sc.fit_transform(new_DataSet2.loc[:,"potential":])

# Creating a new dataFrame for it
sub_set2=pd.DataFrame(scaled,columns=new_DataSet2.loc[:,"potential":].columns)
sub_set2.shape


# In[ ]:


test_data[high_correlations.index]


# In[ ]:


# Using ensembling modelling
from sklearn.ensemble import VotingRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,cross_val_predict
import statsmodels.api as sm
from scipy import stats



# In[ ]:
#training the models using Linear Regression, Random Forest Regression and Gradient Boosting Regression

x = new_DataSet[high_correlations.index].loc[:, "potential":]
y = new_DataSet['overall']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Train and evaluate the models using cross-validation
model_performance = {}
for model_name, model in models.items():
    # Use x_train (lowercase) to match the variable defined earlier
    model.fit(x_train, y_train)
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
    model_performance[model_name] = -cv_scores.mean()

model_performance


# In[ ]:


# a new dataFrame for the scaled dataset with the high correlations dataset
sub_set= pd.DataFrame(scaled,columns=new_DataSet[high_correlations.index].loc[:,"potential":].columns)
sub_set


# In[ ]:


# Fine-tuning the best-performing model 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
y_val_pred = best_model.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
print("Validation MSE:", mse_val)


# In[ ]:


# Extract features and target from the test dataset
X_test = sub_set2
y_test = new_DataSet2['overall']

# Predict and evaluate
y_test_pred = best_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print("Test MSE on New Season Data:", mse_test)


# In[ ]:


# Saving the best model
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)


# In[ ]:


# Save the following code in a file named app.py

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input form
st.title("Player Rating Prediction")

age = st.number_input("Age", min_value=15, max_value=45, value=25)
potential = st.number_input("Potential", min_value=30, max_value=100, value=75)
international_reputation = st.number_input("International Reputation", min_value=1, max_value=5, value=1)
skill_moves = st.number_input("Skill Moves", min_value=1, max_value=5, value=3)
weak_foot = st.number_input("Weak Foot", min_value=1, max_value=5, value=3)
value_eur = st.number_input("Value (in EUR)", min_value=0, max_value=150000000, value=1000000)
wage_eur = st.number_input("Wage (in EUR)", min_value=0, max_value=500000, value=5000)
pace = st.number_input("Pace", min_value=0, max_value=100, value=50)
shooting = st.number_input("Shooting", min_value=0, max_value=100, value=50)
passing = st.number_input("Passing", min_value=0, max_value=100, value=50)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'age': [age],
    'potential': [potential],
    'international_reputation': [international_reputation],
    'skill_moves': [skill_moves],
    'weak_foot': [weak_foot],
    'value_eur': [value_eur],
    'wage_eur': [wage_eur],
    'pace': [pace],
    'shooting': [shooting],
    'passing': [passing]
})

# Button to make predictions
if st.button('Predict Player Rating'):
    prediction = model.predict(input_data)[0]
    st.write(f'The predicted player rating is: {prediction}')


# In[ ]:




