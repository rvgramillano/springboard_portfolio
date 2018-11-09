import pandas as pd
import statsmodels.api as sm 
import pickle
import numpy as np

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'data/LA_df.pkl')

seattle_df = pd.read_pickle(WORKING_DIR + 'data/seattle_df.pkl')


#r^2 seattle: from .419 -> .418
#r^2 LA: from .686 -> .649
def multiple_regression(df, name):
	# still need to work on changing the categorical value 'hourlyskyconditions' to numeric
	#print [col for col in df.columns if col != 'demand' and col!='hourlyskyconditions']
	X = df[[col for col in df.columns if col != 'demand']]
	y = df['demand']
	X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
	
	model = sm.OLS(y, X).fit()
	predictions = model.predict(X)
	
	print('------------------%s------------------'%name)
	print(model.summary())

# with time of day:
#LA: .686 -> .701
#Seattle: .419 -> .587
multiple_regression(la_df, 'LOS ANGELES')
multiple_regression(seattle_df, 'SEATTLE')