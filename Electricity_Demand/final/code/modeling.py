import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

def series_to_supervised(data,  col_names, n_in=1, n_out=1, dropnan=True):
  """
  Frame a time series as a supervised learning dataset.
  Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
  Returns:
    Pandas DataFrame of series framed for supervised learning.
  """
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('%s(t-%d)' % (col_names[j], i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('%s(t)' % (col_names[j])) for j in range(n_vars)]
    else:
      names += [('%s(t+%d)' % (col_names[j], i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

def evaluate(model, X_test, y_test, X_train, y_train, m_name):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Compute and print the metrics
    r2_test = model.score(X_test, y_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    r2_train = model.score(X_train, y_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    print m_name
    
    print '---------------------'
    print 'Train R^2: %.4f' % r2_train
    print 'Train Root MSE: %.4f' % rmse_train

    print '---------------------'
    print 'Test R^2: %.4f' % r2_test
    print 'Test Root MSE: %.4f' % rmse_test

    return r2_test, rmse_test


WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

df = pd.read_pickle(WORKING_DIR + 'data/LA_df.pkl')

#set the column we want to predict (demand) to the first columns for consistency
cols = list(df.columns)
cols.remove('demand')
cols.insert(0,'demand')
df = df[cols]

values = df.values
# ensure all data is float
values = values.astype('float32')
# frame as supervised learning
reframed = series_to_supervised(values, df.columns, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]], axis=1, inplace=True)

values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]


r2 = []
rmse = []
name = []


### LINEAR REGRESSION ###
# Create the model pipeline
steps = [('scaler', MinMaxScaler(feature_range=(0, 1))),
         ('linearregression', LinearRegression())]

pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

scores = cross_val_score(pipeline.steps[1][1], X_test, y_test, cv=5, scoring='r2')
rmse = np.sqrt(-1*cross_val_score(pipeline.steps[1][1], X_test, y_test, cv=5, scoring='neg_mean_squared_error'))


# Evaluate model
r2_score, rmse_score = evaluate(pipeline, X_test, y_test, 'Linear Regression')

r2.append(r2_score)
rmse.append(rmse_score)
name.append('Linear Regression')


### DESCISION TREE REGRESSION ###
# Create the model pipeline
steps = [('scaler', MinMaxScaler(feature_range=(0, 1))),
         ('elasticnet', DecisionTreeRegressor())]

pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Evaluate model
r2_score, rmse_score = evaluate(pipeline, X_test, y_test, 'Decision Tree')

r2.append(r2_score)
rmse.append(rmse_score)
name.append('Decision Tree')


### KNN REGRESSION ###
# Create the model pipeline
steps = [('scaler', MinMaxScaler(feature_range=(0, 1))),
         ('k-NN', KNeighborsRegressor())]

pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Evaluate model
r2_score, rmse_score = evaluate(pipeline, X_test, y_test, 'k-NN')

r2.append(r2_score)
rmse.append(rmse_score)
name.append('k-NN')


### RANDOM FOREST REGRESSION ###
# Create the model pipeline
steps = [('scaler', MinMaxScaler(feature_range=(0, 1))),
         ('randomforest', RandomForestRegressor())]

pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Evaluate model
r2_score, rmse_score = evaluate(pipeline, X_test, y_test, 'Random Forest')

r2.append(r2_score)
rmse.append(rmse_score)
name.append('Random Forest')


### GRADIENT BOOSTING REGRESSION ###
# Create the model pipeline
steps = [('scaler', MinMaxScaler(feature_range=(0, 1))),
         ('gradboost', GradientBoostingRegressor())]

pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Evaluate model
r2_score, rmse_score = evaluate(pipeline, X_test, y_test, 'Gradient Boosting')

r2.append(r2_score)
rmse.append(rmse_score)
name.append('Gradient Boosting')


### BAGGING REGRESSION ###
# Create the model pipeline
steps = [('scaler', MinMaxScaler(feature_range=(0, 1))),
         ('Bagging', BaggingRegressor())]

pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Evaluate model
r2_score, rmse_score = evaluate(pipeline, X_test, y_test, 'Bagging')

r2.append(r2_score)
rmse.append(rmse_score)
name.append('Bagging')



### ADABOOST REGRESSION ###
# Create the model pipeline
steps = [('scaler', MinMaxScaler(feature_range=(0, 1))),
         ('adaboost', AdaBoostRegressor())]

pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Evaluate model
r2_score, rmse_score = evaluate(pipeline, X_test, y_test, 'AdaBoost')

r2.append(r2_score)
rmse.append(rmse_score)
name.append('AdaBoost')

#RESULTS#
la_results = pd.DataFrame({'Model': name, 'R^2': r2, 'RMSE': rmse})
print('-------LA-------')
print(la_results.sort_values(by='R^2', ascending=False))

