import pandas as pd
import statsmodels.api as sm 
import pickle
import numpy as np
import scipy.stats as stats

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'data/LA_df.pkl')

seattle_df = pd.read_pickle(WORKING_DIR + 'data/seattle_df.pkl')

def multiple_regression(df, name):
	X = df[[col for col in df.columns if col != 'demand']]
	y = df['demand']
	X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
	
	model = sm.OLS(y, X).fit()
	predictions = model.predict(X)
	
	print('------------------%s------------------'%name)
	print(model.summary())
	return model

m_la = multiple_regression(la_df, 'LOS ANGELES')
m_seattle = multiple_regression(seattle_df, 'SEATTLE')

# drop high p-value columns and save
la_df = la_df.drop(['hourlywindspeed', 'hourlyheatingdegrees', 'hourlyskyconditions_BKN', 'hourlyskyconditions_FEW', 'hourlyskyconditions_OVC', 'hourlyskyconditions_SCT'], axis=1)
seattle_df = seattle_df.drop(['hourlywindspeed','hourlyvisibility', 'hourlycoolingdegrees', 'hourlyskyconditions_CLR','hourlyskyconditions_BKN', 'hourlyskyconditions_FEW', 'hourlyskyconditions_OVC', 'hourlyskyconditions_SCT'], axis=1)


la_df.to_pickle(WORKING_DIR + 'data/LA_df_final.pkl')
seattle_df.to_pickle(WORKING_DIR + 'data/seattle_df_final.pkl')

