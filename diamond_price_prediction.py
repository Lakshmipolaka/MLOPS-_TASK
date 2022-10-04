#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("C:\\Users\\Lakshmi\\Desktop\\diamonds.csv")
df.head()


# In[ ]:


# reindex the columns of dataset for making dataset more understandable
df = df.reindex(columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price'])
df.head()


# In[ ]:


df.info


# data cleaning

# In[ ]:


df.isnull().sum()


# In[ ]:


df= df.drop_duplicates()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.describe()


# In[ ]:


# all numeric (float and int) variables in the dataset
df_numeric = df.select_dtypes(include=['float64', 'int64'])
df_numeric.head()


# In[ ]:


# correlation matrix
corr = df_numeric.corr()
corr


# In[ ]:


# plotting correlations on a heatmap
plt.figure(figsize=(16,8))
# heatmap
sns.heatmap(corr, annot=True)
plt.show()

Data preparation
# In[ ]:


# Identifying the inputs(x) and output(y)

x = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = df['price']


# In[ ]:


# split into train and test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.75, random_state= 0)


# In[ ]:


x_train


# In[ ]:


# separating train categorical column
x_train_cat = x_train.select_dtypes(include=['object'])
x_train_cat


# In[ ]:


# separating train numerical column
x_train_num = x_train.select_dtypes(include=['int64', 'float64'])
x_train_num

Normalization
# In[ ]:


x_train_cat['cut'].value_counts(normalize=True)


# In[ ]:


x_train_cat['color'].value_counts(normalize=True)


# In[ ]:


x_train_cat['clarity'].value_counts(normalize=True)


# In[ ]:


x_train_cat_le = pd.DataFrame(index= x_train_cat.index)
x_train_cat_le.head()


# In[ ]:


cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}

x_train_cat_le['cut'] = x_train_cat['cut'].apply(lambda k : cut_encoder[k])
x_train_cat_le


# In[ ]:


color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}

x_train_cat_le['color'] = x_train_cat['color'].apply(lambda z : color_encoder[z])
x_train_cat_le


# In[ ]:


clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}

x_train_cat_le['clarity'] = x_train_cat['clarity'].apply(lambda z : clarity_encoder[z])
x_train_cat_le.head()


# In[ ]:


# scaling the numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_num_rescaled = pd.DataFrame(scaler.fit_transform(x_train_num), 
                                    columns = x_train_num.columns, 
                                    index = x_train_num.index)

x_train_num_rescaled.head()


# In[ ]:


x_train_transformed = pd.concat([x_train_num_rescaled, x_train_cat_le], axis=1)
x_train_transformed.head()


# In[ ]:


x_test_cat = x_test.select_dtypes(include=['object'])
x_test_cat


# In[ ]:


x_test_num = x_test.select_dtypes(include=['int64', 'float64'])
x_test_num.head()


# In[ ]:


x_test_num_rescaled = pd.DataFrame(scaler.transform(x_test_num), 
                                   columns = x_test_num.columns, 
                                   index = x_test_num.index)

x_test_num_rescaled.head()


# In[ ]:


x_test_cat_le = pd.DataFrame(index = x_test_cat.index)
x_test_cat_le.head()


# In[ ]:


x_test_cat_le['cut'] = x_test_cat['cut'].apply(lambda z : cut_encoder[z])
x_test_cat_le['color'] = x_test_cat['color'].apply(lambda z : color_encoder[z])
x_test_cat_le['clarity'] = x_test_cat['clarity'].apply(lambda z: clarity_encoder[z])

x_test_cat_le.head()


# In[ ]:


x_test_transformed = pd.concat([x_test_num_rescaled, x_test_cat_le], axis=1)
x_test_transformed.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_new = pd.DataFrame(scaler.fit_transform(x_train_transformed),columns = x_train_transformed.columns,index = x_train_transformed.index)
x_train_new.head()


# In[ ]:


x_test_new = pd.DataFrame(scaler.fit_transform(x_test_transformed),columns = x_test_transformed.columns,index = x_test_transformed.index)
x_test_new.head()


# In[ ]:


print("X_train:",x_train_new.shape,"X_test:" ,x_test_new.shape,"y_train:",y_train.shape,"y_test:",y_test.shape)


# ML flow integration

# In[ ]:


import mlflow


# In[ ]:


mlflow.set_tracking_uri("sqlite:///test_mlflow.db")


# In[ ]:


mlflow.set_experiment("Diamond Price Prediction")


# In[ ]:


from pickle import dump

dump(scaler, open('standard_scaler.pkl', 'wb'))


# In[ ]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# In[ ]:


with mlflow.start_run():
    mlflow.set_tag("dev", "lakshmipolaka")
    mlflow.set_tag("algo", "Linear Regression")
    mlflow.log_param("data-path", "diamonds.csv")
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train_new, y_train)
    y_test_pred = linear_regressor.predict(x_test_new)
    acc = metrics.r2_score(y_test, y_test_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(linear_regressor, artifact_path = "models")
    mlflow.log_artifact("standard_scaler.pkl")


# In[ ]:


with mlflow.start_run():
    mlflow.set_tag("dev", "lakshmipolaka")
    mlflow.set_tag("algo", "KNN")
    mlflow.log_param("data-path", "diamonds.csv")
    k = 9
    mlflow.log_param("n_neighbors", k)
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(x_train_new, y_train)
    y_test_pred = knn_regressor.predict(x_test_new)
    r2score_acc_knn = metrics.r2_score(y_test, y_test_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    mlflow.log_metric("accuracy", r2score_acc_knn)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(knn_regressor, artifact_path="models")
    mlflow.log_artifact("standard_scaler.pkl")


# In[ ]:


with mlflow.start_run():
    mlflow.set_tag("dev", "lakshmipolaka")
    mlflow.set_tag("algo", "Random Forest Regression")
    mlflow.log_param("data-path", "diamonds.csv")
    t = 20
    rf_regressor = RandomForestRegressor(n_estimators = t)
    rf_regressor.fit(x_train_new, y_train)
    y_test_pred = rf_regressor.predict(x_test_new)
    acc = metrics.r2_score(y_test, y_test_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    mlflow.log_param("n_estimators", t)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(rf_regressor, artifact_path = "models")
    mlflow.log_artifact("standard_scaler.pkl")


# In[ ]:


with mlflow.start_run():
    mlflow.set_tag("dev", "lakshmipolaka")
    mlflow.set_tag("algo", "Decision Tree Regression")
    mlflow.log_param("data-path", "diamonds.csv")
    d = None
    dt_regressor = DecisionTreeRegressor(max_depth = d)
    dt_regressor.fit(x_train_new, y_train)
    y_test_pred = dt_regressor.predict(x_test_new)
    acc = metrics.r2_score(y_test, y_test_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    mlflow.log_param("max_depth", d)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(dt_regressor, artifact_path = "models")
    mlflow.log_artifact("standard_scaler.pkl")


# Experiment Run 5 - KNN with Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Enabling automatic MLflow logging for scikit-learn runs
mlflow.sklearn.autolog(max_tuning_runs=None)

with mlflow.start_run():
    tuned_parameters = [{'n_neighbors': [i for i in range(1, 51)], 'p': [1, 2]}]
    gs = GridSearchCV(
        estimator = KNeighborsRegressor(),
        param_grid = tuned_parameters,
        scoring = 'neg_mean_squared_error',
        cv = 5,
        return_train_score = True,
        verbose = 1
    )
    gs.fit(x_train_new, y_train)
    mlflow.sklearn.autolog(disable = True)


# In[ ]:





# In[ ]:




