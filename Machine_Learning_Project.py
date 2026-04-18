#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Get to know the data.


# In[2]:


# Import all necessary libraries.
### All comments with three number signs are just for the grader to understand my thought process and will not be what I post to my GitHub.  After it's been graded and I know it's good enough for my portfolio I will delete these comments.
### I attempted to import all of the libaries from this class so that if I wanted to look at something during the preprocessing stage anything would be available. 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from mord import LogisticIT
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score
from dmba import regressionSummary
from sklearn.model_selection import KFold, cross_val_score


# In[3]:


# I choose to work with the bike dataset in days instead of hours to look at overall demand.
bike_df = pd.read_csv("day.csv")


# In[4]:


# Look at the dataset.
bike_df.head()


# In[5]:


# Get a better understanding of hte dimensions. 
print(bike_df)


# In[6]:


# Start preprocessing the data.


# In[7]:


# Getting a list of columns all together can help make decisions. 
bike_df.columns


# In[8]:


# Check to see if you have any missing values.
nan_check_bike_df = bike_df.isna()
print(nan_check_bike_df)


# In[9]:


# Get a count of missing values.
has_nan = bike_df.isna().any().any()
print(f"Number of nan values: {has_nan}")


# In[10]:


# Look at the columns one more time so you can begin your preprocessing.
bike_df.columns


# In[11]:


# Make sure each columns' variables are the right kind that we can work with.
bike_df["season"] = bike_df["season"].astype("category")
bike_df["mnth"] = bike_df["mnth"].astype("category")
bike_df["holiday"] = bike_df["holiday"].astype("category")
bike_df["weekday"] = bike_df["weekday"].astype("category")
bike_df["weathersit"] = bike_df["weathersit"].astype("category")
bike_df["temp"] = bike_df["temp"].astype("float")
bike_df["atemp"] = bike_df["atemp"].astype("float")
bike_df["hum"] = bike_df["hum"].astype("float")
bike_df["windspeed"] = bike_df["windspeed"].astype("float")
bike_df["cnt"] = bike_df["cnt"].astype("int")


# In[12]:


# Drop unneccessary columns. We want to know total count to be able to predict demand so we don't need categories, just total count.
bike_df = bike_df.drop(["casual", "registered", "instant"], axis = 1)


# In[13]:


# Check on the changes you made.
bike_df.head()


# In[14]:


# Use MinMax scaler to scale numerical variables.
scaler = MinMaxScaler()
columns_to_scale = ["temp", "atemp", "hum", "windspeed", "cnt"]
bike_norm_df = pd.DataFrame(
    scaler.fit_transform(bike_df[columns_to_scale]),
    index=bike_df.index,
    columns=columns_to_scale
)


# In[15]:


# Make sure it worked.
bike_df.head()


# In[16]:


# Now we want to standardize the numerical values so that we can also make calculations with categorical varaibles.
scaler = StandardScaler()
bike_df_norm = pd.DataFrame(scaler.fit_transform(bike_df[columns_to_scale]))


# In[17]:


# Look at the results.
bike_df_norm.head()


# In[18]:


# Join the scaled data to the orginal DataFrame.
bike_df = bike_df.join(bike_df_norm)


# In[19]:


# Check on the results
bike_df.head()


# In[20]:


# Drop the unscaled columns.
bike_df = bike_df.drop(["temp", "atemp", "hum", "windspeed", "cnt"], axis = 1)


# In[21]:


# Check the results.
bike_df.head()


# In[22]:


# Rename the scaled columns to original names so it's easy to understand calculations.
bike_df = bike_df.rename(columns = {0 : "temp", 1 : "atemp", 2 : "hum", 3: "windspeed", 4 : "cnt"})


# In[23]:


# Check results.
bike_df.head()


# In[24]:


# Take another look to get a better grasp of your data.
bike_df.tail()


# In[25]:


categorical_vars = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

for var in categorical_vars:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=bike_df, x=var, y='cnt')
    plt.title(f'Total Bike Count by {var.capitalize()}')
    plt.xlabel(var.capitalize())
    plt.ylabel('Total Bike Count')
    plt.tight_layout()
    plt.show()


# In[26]:


# Make a variable that holds all of your categorical data.
non_ordinal_cats = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]


# In[27]:


# Now get all of the dummies at once.
dummies = pd.get_dummies(
    bike_df[non_ordinal_cats],
    prefix_sep="-",
    drop_first=True 
)


# In[28]:


# Replace the categorical columns that haven't been processed with the new ones you got dummies for.
bike_df_encoded = pd.concat([
    bike_df.drop(columns=non_ordinal_cats),
    dummies
], axis=1)


# In[29]:


# Make sure everything worked.
bike_df_encoded.head()


# In[30]:


# Change the name back to bike_df to make things easier for yourself.
bike_df = bike_df_encoded


# In[31]:


# Make sure it worked.
bike_df.head()


# In[32]:


continuous_vars = ["temp", "atemp", "hum", "windspeed"]

for var in continuous_vars:
    plt.figure(figsize=(8, 5))
    sns.regplot(data = bike_df, x = var, y = "cnt", scatter_kws={"alpha" : 0.5}, line_kws={"color":"red"})
    plt.title(f"Total Bike Count vs {var.capitalize()}")
    plt.xlabel(var.capitalize())
    plt.ylabel('Total Bike Count')
    plt.tight_layout()
    plt.savefig(f'regression_{var}.png', bbox_inches='tight')
    plt.show()


# In[33]:


# Multiple Linear Regression


# In[34]:


# Partition the data.
predictors = ["temp", "hum", "windspeed", "workingday"]
outcome = "cnt"
X = bike_df[predictors]
y = bike_df[outcome]


# In[35]:


train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.4, random_state = 1)


# In[36]:


# Run Linear Regression
bike_df = LinearRegression()
bike_df.fit(train_X, train_y)


# In[37]:


# Look at the coefficients.
print(pd.DataFrame({"Predictor" : X.columns, "coefficient" : bike_df.coef_}))


# In[38]:


# Look a the Regression statistics.
regressionSummary(train_y, bike_df.predict(train_X))


# In[39]:


# use predict() to make predictions on the validation set.
bike_df_pred = bike_df.predict(valid_X)
result = pd.DataFrame({"Predicted" : bike_df_pred, "Actual" : valid_y, "Residual" : valid_y - bike_df_pred})
print(result.head(20))


# In[40]:


# Print performance measures.
regressionSummary(valid_y, bike_df.predict(valid_X))


# In[41]:


# Check the shape of train_X and valid_X.
print("train_X shape:" , train_X.shape)
print("valid_X shape:" , valid_X.shape)


# In[42]:


# Run Lasso regression.
lasso = Lasso(alpha = 1)
lasso.fit(train_X, train_y)
regressionSummary(valid_y, lasso.predict(valid_X)) # I did not Standardize the data again because the results were exactly the same when I did.


# In[43]:


# Run Ridge Regression.
ridge = Ridge(alpha = 1)
ridge.fit(train_X, train_y)
regressionSummary(valid_y, ridge.predict(valid_X))


# In[44]:


# Run Bayesian Ridge Regression.
bayesianRidge = BayesianRidge()
bayesianRidge.fit(train_X, train_y)
regressionSummary(valid_y, bayesianRidge.predict(valid_X))


# In[45]:


# Create the model - Lasso.
lasso_model = Lasso(alpha = 1.0)


# In[46]:


# Define the k-fold cross validator.
k = 5
kf = KFold(n_splits = k, shuffle = True, random_state = 42)


# In[47]:


# Perform k-fold cross-validation.
lasso_scores = cross_val_score(lasso_model, X, y, cv = kf, scoring = "neg_mean_squared_error")


# In[48]:


# Convert the scores to positive values and take the square root to get RMSE.
lasso_rmse_scores = np.sqrt(-lasso_scores)


# In[49]:


# Print the results.
print(f"Lasso RMSE scores for each fold : {lasso_rmse_scores}")
print(f"Average Lasso RMSE : {lasso_rmse_scores.mean()}")
print(f"Standard Deviation of Lasso RMSE : {lasso_rmse_scores.std()}")


# In[50]:


# Create the model - Ridge
ridge_model = Ridge(alpha = 1.0)


# In[51]:


#Define the k-fold cross-validator.
k = 5
kf = KFold(n_splits = k, shuffle = True, random_state = 42)


# In[52]:


# Perform k-fold cross-validation.
ridge_scores = cross_val_score(ridge_model, X, y, cv = kf, scoring = "neg_mean_squared_error")


# In[53]:


# Convert the scores to positive values and tkae the square root to get RMSE.
ridge_rmse_scores = np.sqrt(-ridge_scores)


# In[54]:


# Print the results.
print(f"Ridge RMSE scores for each fold: {ridge_rmse_scores}")
print(f"Average Ridge RMSE: {ridge_rmse_scores.mean()}")
print(f"Standard Deviation of Ridge RMSE: {ridge_rmse_scores.std()}")


# In[55]:


# Make a scatter plot to visualize the Ridge Regression since it was the most accurate.
y_pred = ridge.predict(valid_X)

plt.scatter(valid_y, y_pred, alpha=1.0)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression: Actual vs Predicted")

# Reference line: perfect prediction
min_val = min(valid_y.min(), y_pred.min())
max_val = max(valid_y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.savefig("ridge_actual_vs_predicted.png", bbox_inches="tight")
plt.show()


# In[56]:


# Time Series Analysis


# In[57]:


# Reload the dataset to ensure bike_df is a DataFrame.
bike_df = pd.read_csv("day.csv")

# Convert "dteday" to datetime format.
bike_df["dteday"] = pd.to_datetime(bike_df["dteday"])

# Set "dteday" as the index.
bike_df.set_index("dteday", inplace=True)

# Set frequency for time series analysis.
bike_df = bike_df.asfreq("D")

# Parse the target variable as a time series.
bike_series = bike_df["cnt"]


# In[58]:


# Define a finction to calculate error metrics.
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mape, mae, mse, rmse


# In[59]:


# Extend the index to include the next 365 days.
future_dates = pd.date_range(start=bike_series.index[-1] + pd.offsets.Day(), periods=365, freq="D")


# In[60]:


# 7-Day Moving Average
### I chose 7 because they are 7 days in a week which would show us the fluctuations by week.
moving_avg_7 = bike_series.rolling(window = 7).mean()


# In[61]:


# Look that results visually by plotting.
plt.figure(figsize=(12, 6))
plt.plot(bike_series, label="Daily count", alpha=0.4)
plt.plot(moving_avg_7, label="7-day Moving Avg", color="red")
plt.legend()
plt.title("Bike Rentals and 7-Day Moving Average")
plt.savefig("bike_rentals_moving_avg.png", bbox_inches="tight")
plt.show()


# In[62]:


# Make predictions for the next 7 days.
moving_avg_7_future = []
extended_series = bike_series.copy()
for _ in range(7):
    next_value = extended_series[-7:].mean()
    moving_avg_7_future.append(next_value)
    extended_series = pd.concat([extended_series, pd.Series([next_value], index = [extended_series.index[-1] + pd.offsets.Day()])])


# In[63]:


# Print the results
for i, prediction in enumerate(moving_avg_7_future, 1):
    print(f"Day {i} prediction {prediction: .2f}")


# In[64]:


# 30-Day Moving Average
### I chose 30 because they are typically 30 days in a month which would roughly show us the fluctuations by year.
moving_avg_30 = bike_series.rolling(window = 30).mean()


# In[65]:


# Look the results visually.
plt.figure(figsize=(12, 6))
plt.plot(bike_series, label="Daily count", alpha=0.4)
plt.plot(moving_avg_30, label="30-day Moving Avg", color="red")
plt.legend()
plt.title("Bike Rentals and 30-Day Moving Average")
plt.show()


# In[66]:


# Make predictions for the next 30 days.
moving_avg_30_future = []
extended_series = bike_series.copy()
for _ in range(7):
    next_value = extended_series[-30:].mean()
    moving_avg_30_future.append(next_value)
    extended_series = pd.concat([extended_series, pd.Series([next_value], index = [extended_series.index[-1] + pd.offsets.Day()])])


# In[67]:


# Print the results.
for i, prediction in enumerate(moving_avg_30_future, 1):
    print(f"Day {i} prediction {prediction: .2f}")


# In[68]:


# Weighted 7 Day Moving Average without Lambda Function.
weighted_moving_avg_7_future = []
extended_series = bike_series.copy() # It's a good idea to not change original dataset.

weights = np.array([0.5, 0.3, 0.2])
weights = weights / weights.sum()  # Normalize weights

for _ in range(7):
    if len(extended_series) < len(weights):
        next_value = extended_series.mean()  # this handles missing values just in case I missed them
    else:
        recent_values = extended_series[-len(weights):]
        next_value = np.dot(recent_values.values[::-1], weights) ### The original code that we learned in class was not working, so I did some research and found the dot function.  This grabs the most recent and reverses the order and performs a dot product calculation on them.  A dot product combines two vectors to produce a single scaler vector, essentially a differnt way of standardizing the values.
    
    weighted_moving_avg_7_future.append(next_value)
    next_index = extended_series.index[-1] + pd.offsets.Day()
    extended_series = pd.concat([extended_series, pd.Series([next_value], index=[next_index])])
### The code from the book was not working so I did some research and found these work arounds.


# In[69]:


# Print the results.
for i, prediction in enumerate(weighted_moving_avg_7_future, 1):
    print(f"Day {i} weighted prediction : {prediction: .2f}")


# In[70]:


# Exponential Smoothing with smoothing constant 0.2 using Holt.
exp_smoothing_0_2_model = Holt(bike_series, initialization_method = "estimated").fit(smoothing_level = 0.2)
exp_smoothing_0_2 = exp_smoothing_0_2_model.fittedvalues
exp_smoothing_0_2_future = exp_smoothing_0_2_model.forecast(30)


# In[71]:


# Print the results.
for i, prediction in enumerate(exp_smoothing_0_2_future, 1):
    print(f"Day {i} exponential smoothing preduction: {prediction : .2f}")


# In[72]:


# Simple Linear Regression - just to see what other options we have.
days = np.arange(len(bike_series)).reshape(-1, 1)
reg_model = LinearRegression().fit(days, bike_series)
future_days = np.arange(len(bike_series), len(bike_series) + 6).reshape(-1, 1)
linear_regression = reg_model.predict(days)
linear_regression_future = reg_model.predict(future_days)
linear_regression_series = pd.Series(linear_regression, index=bike_series.index)


# In[73]:


# Print the reuslts.
for i, prediction in enumerate(linear_regression_future, 1):
    print(f"Month {i} linear regression prediction: {prediction : .2f}")


# In[74]:


### The original calculate_metrics function did not work, so I reset the indexes to match with the actual and was able to get the results.
def calculate_metrics(actual, forecast):
    common_index = actual.index.intersection(forecast.index)
    actual_aligned = actual.loc[common_index]
    forecast_aligned = forecast.loc[common_index]

    mae = mean_absolute_error(actual_aligned, forecast_aligned)
    mse = mean_squared_error(actual_aligned, forecast_aligned)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100
    return mape, mae, mse, rmse
# Create a Series with the correct index for the last 7 days
forecast_series_7 = pd.Series(weighted_moving_avg_7_future, index=bike_series.index[-7:])

# Then use it in your metrics dictionary
metric = {
    "7-Day Moving Average": calculate_metrics(bike_series, moving_avg_7.dropna()),
    "30-Day Moving Average": calculate_metrics(bike_series, moving_avg_30.dropna()),
    "7-Day Weighted Moving Average": calculate_metrics(bike_series[-7:], forecast_series_7),
    "Exponential Smoothing 0.2": calculate_metrics(bike_series, exp_smoothing_0_2),
    "Linear Regression": calculate_metrics(bike_series, linear_regression_series)
}


# In[75]:


# Put the results into a DataFrame and print them.
metrics_df = pd.DataFrame(metric, index = ["MAPE", "MAE", "MSE", "RMSE"])
metrics_df


# In[ ]:




