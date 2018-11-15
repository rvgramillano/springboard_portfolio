import pickle
import pandas as pd

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'data/LA_df_first.pkl')
la_df = la_df.drop(['hourlywetbulbtempf', 'hourlydrybulbtempf', 'hourlyaltimetersetting', 'hourlysealevelpressure'], axis=1)
#0 for daytime(6H, 18H), 1 for nighttime
la_df['hourlytimeofday'] = [0 if (hour >= 6 and hour <= 18) else 1 for hour in la_df.index.hour]
la_df = pd.get_dummies(la_df)
la_df = la_df.drop(['hourlyskyconditions_VV'], axis=1)

seattle_df = pd.read_pickle(WORKING_DIR + 'data/seattle_df_first.pkl')
seattle_df = seattle_df.drop(['hourlywetbulbtempf', 'hourlydrybulbtempf', 'hourlyaltimetersetting', 'hourlysealevelpressure'], axis=1)
seattle_df['hourlytimeofday'] = [0 if (hour >= 6 and hour <= 18) else 1 for hour in seattle_df.index.hour]
seattle_df = pd.get_dummies(seattle_df)
seattle_df = seattle_df.drop(['hourlyskyconditions_VV'], axis=1)

la_df.to_pickle(WORKING_DIR + 'data/LA_df.pkl')
seattle_df.to_pickle(WORKING_DIR + 'data/seattle_df.pkl')