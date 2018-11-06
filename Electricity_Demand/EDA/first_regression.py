import pandas as pd
import statsmodels.api as sm 
import pickle
import numpy as np

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'LA_df.pkl')
seattle_df = pd.read_pickle(WORKING_DIR + 'seattle_df.pkl')

def multiple_regression(df, name):
	# still need to work on changing the categorical value 'hourlyskyconditions' to numeric
	print [col for col in df.columns if col != 'demand' and col!='hourlyskyconditions']
	X = df[[col for col in df.columns if col != 'demand' and col!='hourlyskyconditions']]
	y = df['demand']
	#X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
	
	model = sm.OLS(y, X).fit()
	predictions = model.predict(X)
	
	print('------------------%s------------------'%name)
	print(model.summary())


multiple_regression(la_df, 'LOS ANGELES')
multiple_regression(seattle_df, 'SEATTLE')