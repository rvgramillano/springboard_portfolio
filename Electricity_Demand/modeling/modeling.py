import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.stats import randint

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'data/LA_df.pkl')

seattle_df = pd.read_pickle(WORKING_DIR + 'data/seattle_df.pkl')

df = la_df.copy()

y = df[['demand']]
X = df.drop(['demand'], axis=1)

def evaluate(model, X, y, X_test, y_test, m_name):
	#y_pred_prob = model.predict_proba(X_test)[:,1]
	y_pred = model.predict(X_test)

	# Compute and print the metrics
	r2 = model.score(X_test, y_test)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))

	# Generate ROC curve values: fpr, tpr, thresholds
	#fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

	#auc_score = roc_auc_score(y_test, y_pred_prob)

	#metrics
	adj_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)

	# Plot ROC curve
	#plt.plot([0, 1], [0, 1], 'k--')
	#plt.plot(fpr, tpr)
	#plt.xlabel('False Positive Rate')
	#plt.ylabel('True Positive Rate')
	#plt.title('ROC Curve for %s' % m_name)
	#plt.save_fig(WORKING_DIR + 'plots/modeling/%s_roc.png' % m_name, dpi=300)
	#plt.show()
	#raw_input('...')
	#plt.close()
	print m_name
	print '---------------------'
	print 'R^2: %.4f' % r2
	print 'adj R^2: %.4f' % adj_r2
	print 'Root MSE: %.4f' % rmse

	return r2, adj_r2, rmse


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)

### LINEAR REGRESSION ###
t_start = time.time()
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('linearregression', LinearRegression())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Fit to the training set
pipeline.fit(X_train, y_train)

r2, adj_r2, mse = evaluate(pipeline, X, y, X_test, y_test, 'Linear Regression')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')






### ELASTIC NET REGRESSION ###
t_start = time.time()
#alpha=1
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30),'elasticnet__alpha':np.linspace(.1,10,10)}

# Create the RandomizedSearchCV object: rm_cv
gm_cv = GridSearchCV(pipeline, parameters, cv=5)

# Fit to the training set
gm_cv.fit(X_train,y_train)

m = gm_cv.best_estimator_
print(gm_cv.best_params_)

r2, adj_r2, mse = evaluate(m, X, y, X_test, y_test, 'ElasticNet')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')






### DESCISION TREE REGRESSION ###
t_start = time.time()
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('DecisionTreeRegressor', DecisionTreeRegressor())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

parameters = {"DecisionTreeRegressor__max_depth": [3, None],
              "DecisionTreeRegressor__max_features": randint(1, X.shape[1]),
              "DecisionTreeRegressor__min_samples_leaf": randint(1, 9),
              "DecisionTreeRegressor__criterion": ["mae", "mse"]}

# Create the RandomizedSearchCV object: rm_cv
rm_cv = RandomizedSearchCV(pipeline, parameters, cv=5)

rm_cv.fit(X_train,y_train)

m = rm_cv.best_estimator_
print(rm_cv.best_params_)

r2, adj_r2, mse = evaluate(m, X, y, X_test, y_test, 'Decision Tree')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')




### KNN REGRESSION ###
##long time to hypertune, even with random search##
print('default settings')
t_start = time.time()
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('KNeighborsRegressor', KNeighborsRegressor())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

parameters = {"KNeighborsRegressor__n_neighbors": np.arange(3,6)}#,
              #"KNeighborsRegressor__weights": ['uniform', 'distance'],
              #"KNeighborsRegressor__leaf_size": randint(30,60),
              #"KNeighborsRegressor__metric": ["minkowski", "euclidean", 'manhattan']}

# Create the GridSearchCV object: rm_cv
#gm_cv = GridSearchCV(pipeline, parameters, cv=5)

#gm_cv.fit(X_train,y_train)
pipeline.fit(X_train,y_train)

#m = gm_cv.best_estimator_
#print(gm_cv.best_params_)

r2, adj_r2, mse = evaluate(pipeline, X, y, X_test, y_test, 'k-NN')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')



### RANDOM FOREST REGRESSION ###
##long time to hypertune, even with random search##
print('default settings')
t_start = time.time()
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('RandomForestRegressor', RandomForestRegressor())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

parameters = {"RandomForestRegressor__n_estimators": randint(10,51),
              "RandomForestRegressor__max_depth": [3, None],
              "RandomForestRegressor__min_samples_leaf": randint(1, 9),
              "RandomForestRegressor__criterion": ["mae", "mse"],
              "RandomForestRegressor__max_features": randint(1, X.shape[1])}

# Create the RandomizedSearchCV object: rm_cv
#rm_cv = RandomizedSearchCV(pipeline, parameters, cv=5)

#rm_cv.fit(X_train,y_train)
pipeline.fit(X_train,y_train)

#m = rm_cv.best_estimator_

r2, adj_r2, mse = evaluate(pipeline, X, y, X_test, y_test, 'Random Forest')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')






### GRADIENT BOOSTING REGRESSION ###
t_start = time.time()
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('GradientBoostingRegressor', GradientBoostingRegressor())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

parameters = {"RandomForestRegressor__n_estimators": randint(10,51),
              "RandomForestRegressor__max_depth": [3, None],
              "RandomForestRegressor__min_samples_leaf": randint(1, 9),
              "RandomForestRegressor__criterion": ["mae", "mse"],
              "RandomForestRegressor__max_features": randint(1, X.shape[1])}

# Create the RandomizedSearchCV object: rm_cv
#rm_cv = RandomizedSearchCV(pipeline, parameters, cv=5)

#rm_cv.fit(X_train,y_train)
pipeline.fit(X_train,y_train)

#m = rm_cv.best_estimator_

r2, adj_r2, mse = evaluate(pipeline, X, y, X_test, y_test, 'Gradient Boosting')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')








### BAGGING REGRESSION ###
t_start = time.time()
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('BaggingRegressor', BaggingRegressor())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

parameters = {"RandomForestRegressor__n_estimators": randint(10,51),
              "RandomForestRegressor__max_depth": [3, None],
              "RandomForestRegressor__min_samples_leaf": randint(1, 9),
              "RandomForestRegressor__criterion": ["mae", "mse"],
              "RandomForestRegressor__max_features": randint(1, X.shape[1])}

# Create the RandomizedSearchCV object: rm_cv
#rm_cv = RandomizedSearchCV(pipeline, parameters, cv=5)

#rm_cv.fit(X_train,y_train)
pipeline.fit(X_train,y_train)

#m = rm_cv.best_estimator_

r2, adj_r2, mse = evaluate(pipeline, X, y, X_test, y_test, 'Bagging')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')









### ADABOOST REGRESSION ###
t_start = time.time()
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('AdaBoostRegressor', AdaBoostRegressor())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

parameters = {"RandomForestRegressor__n_estimators": randint(10,51),
              "RandomForestRegressor__max_depth": [3, None],
              "RandomForestRegressor__min_samples_leaf": randint(1, 9),
              "RandomForestRegressor__criterion": ["mae", "mse"],
              "RandomForestRegressor__max_features": randint(1, X.shape[1])}

# Create the RandomizedSearchCV object: rm_cv
#rm_cv = RandomizedSearchCV(pipeline, parameters, cv=5)

#rm_cv.fit(X_train,y_train)
pipeline.fit(X_train,y_train)

#m = rm_cv.best_estimator_

r2, adj_r2, mse = evaluate(pipeline, X, y, X_test, y_test, 'AdaBoost')
print('time elapsed = %.2f sec' % (time.time() - t_start) )
print('\n')

