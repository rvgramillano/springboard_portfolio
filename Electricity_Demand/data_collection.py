import pandas as pd
from datetime import datetime
import requests
import numpy as np

EIA_API = ''

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

def EIA_request_to_df(req, value_name):
	'''
	This function unpacks the JSON file into a pandas dataframe.'''
	dat = req['series'][0]['data']
	dates = []
	values = []
	for date, value in dat:
		dates.append(date)
		values.append(float(value))
	df = pd.DataFrame({'date': dates, value_name: values})
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	return df


# collect electricty data for Los Angeles
REGION_CODE = 'LDWP'

# megawatthours
url_demand = requests.get('http://api.eia.gov/series/?api_key=%s&series_id=EBA.%s-ALL.D.H' % (EIA_API, REGION_CODE)).json()
demand_df = EIA_request_to_df(url_demand, 'demand')

# megawatthours
url_generation = requests.get('http://api.eia.gov/series/?api_key=%s&series_id=EBA.%s-ALL.NG.H' % (EIA_API, REGION_CODE)).json()
generation_df = EIA_request_to_df(url_generation, 'generation')

# megawatthours
url_interchange = requests.get('http://api.eia.gov/series/?api_key=%s&series_id=EBA.%s-ALL.TI.H' % (EIA_API, REGION_CODE)).json()
interchange_df = EIA_request_to_df(url_interchange, 'interchange')

# merge dataframes to get all electricity data together
electricity_df = demand_df.merge(generation_df, right_index=True, left_index=True, how='outer').merge(interchange_df, right_index=True, left_index=True, how='outer')

# clean electricity_df of outlier values. this cut removes ~.01% of the data
electricity_df = electricity_df[electricity_df['demand'] != 0]
electricity_df = electricity_df[electricity_df['generation'] > 0]


def fix_date(df):
    '''
    This function goes through the dates in the weather dataframe and if there is more than one record for each
    hour, we pick the record closest to the hour and drop the rows with the remaining records for that hour.

    This is so we can align this dataframe with the one containing electricity data.'''
    new_dates = []
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    # we append the first value to fix indexing
    new_dates.append(df['date'][0].replace(minute=00))
    hour_dict = {}
    keys_drop = []
    for j,date in enumerate(df['date']):
        current_hour = date.hour
        # skip the first value
        if j == 0: 
            continue
        # once we reach the end, just append the last value with minutes replaced to the hour
        if j == len(df['date'])-1:
            new_dates.append(date.replace(minute=00))
            continue
        # if the current hour we're on has another entry in the same hour (the one after, since they are ordered 
        # chronologically), add this date to a dictionary
        if current_hour == df['date'][j+1].hour:
            hour_dict[date] = j
        # once there are no more entries for the same hour
        # we pick the entry closest to the hours and record the index of the rows to drop
        elif len(hour_dict) >= 1:
            hour_dict[date] = j
            min_date = min(hour_dict.keys())
            new_dates.append(min_date.replace(minute=00))
            del hour_dict[min_date]
            keys_del = hour_dict.values()
            keys_drop.extend(keys_del)
            hour_dict = {}
        # for the base case, when there is only one entry per hour
        else:
            new_dates.append(date.replace(minute=00))
    # drop rows designated before (multiple values in a single hour)
    # and relabel the date column and set it as the index
    df_return = df.drop(keys_drop)
    df_return['date'] = new_dates
    df_return = df_return.set_index('date')
    return df_return


def clean_sky_condition(df):
	'''
	This function cleans the hourly sky condition column by assigning the hourly sky condition to be the one at the
	top cloud layer, which is the best determination of the sky condition, as described by the documentation.'''
	conditions = df['hourlyskyconditions']
	new_condition = []
	for k, condition in enumerate(conditions):
		if type(condition) != str and np.isnan(condition):
			new_condition.append(np.nan)
		else:
			colon_indices = [i for i, char in enumerate(condition) if char == ':']
			n_layers = len(colon_indices)
			colon_position = colon_indices[n_layers - 1]
			if condition[colon_position - 1] == 'V':
				condition_code = condition[colon_position - 2 : colon_position]
			else:
				condition_code = condition[colon_position - 3 : colon_position]
			new_condition.append(condition_code)
	df['hourlyskyconditions'] = new_condition
	df['hourlyskyconditions'] = df['hourlyskyconditions'].astype('category')
	return df

def hourly_degree_days(df):
	'''
	This function adds hourly heating and cooling degree days to the weather DataFrame.'''
	df['hourlycoolingdegrees'] = df['hourlywetbulbtempf'].apply(lambda x: x - 65. if x >= 65. else 0.)
	df['hourlyheatingdegrees'] = df['hourlywetbulbtempf'].apply(lambda x: 65. - x if x <= 65. else 0.)

	return df

# collect weather data for los angeles

weather_df = pd.read_csv(WORKING_DIR + 'LA_weather.csv')
# make columns lowercase for easier access
weather_df.columns = [col.lower() for col in weather_df.columns]
# make list of non-hourly columns to keep
daily_columns_keep = ['heatingdegreedays', 'coolingdegreedays']
daily_columns_keep = ['daily' + col for col in daily_columns_keep]

# fixing dataframe so that there's only one record per hour
weather_df = fix_date(weather_df)


for column in weather_df.columns:
	# keep hourly fahrenheit columns
    if column.startswith('hourly') and column.endswith('c'):
        weather_df = weather_df.drop(column, axis=1)
    elif column in daily_columns_keep or column.startswith('hourly'):
        continue
    else:
    	weather_df = weather_df.drop(column, axis=1)

# remove aggregate pressure columns since they contain many null values and hourly values for pressure are already available
weather_df = weather_df.drop('hourlypressurechange', axis=1)
weather_df = weather_df.drop('hourlypressuretendency', axis=1)
# too many null values and already have wind speed measurments
weather_df = weather_df.drop('hourlywindgustspeed', axis=1)
# too many nulls and information too disperse to make meaningful predictions
weather_df = weather_df.drop('hourlyprsentweathertype', axis=1)
# column has no predictive power regarding electicity generation
weather_df = weather_df.drop('hourlywinddirection', axis=1)

# fill the daily heating and cooling degree days such that each hour in an individual day has the same value
weather_df['dailyheatingdegreedays'] = weather_df['dailyheatingdegreedays'].bfill()
weather_df['dailycoolingdegreedays'] = weather_df['dailycoolingdegreedays'].bfill()

weather_df = clean_sky_condition(weather_df)

# clean other columns by replacing string based values with floats
# values with an 's' following indicate uncertain measurments. we simply change those to floats and include them like normal
weather_df['hourlyvisibility'] = weather_df['hourlyvisibility'].apply(lambda x: float(x) if str(x)[-1] != 'V' else float(str(x)[:-1]))

weather_df['hourlydrybulbtempf'] = weather_df['hourlydrybulbtempf'].apply(lambda x: float(x) if str(x)[-1] != 's' else float(str(x)[:-1]))

weather_df['hourlydewpointtempf'] = weather_df['hourlydewpointtempf'].apply(lambda x: float(x) if str(x)[-1] != 's' else float(str(x)[:-1]))

# set trace amounts equal to zero
weather_df['hourlyprecip'][weather_df['hourlyprecip'] == 'T'] = 0.0
weather_df['hourlyprecip'] = weather_df['hourlyprecip'].apply(lambda x: float(x) if str(x)[-1] != 's' else float(str(x)[:-1]))

weather_df = hourly_degree_days(weather_df)


'''
in the following section I plot distributions of all the features to determine what columns should be filled by using the median
and which should be filled according to ffill. the features whose medians and means are close together suggest few outliers
and that the median is a good choice for NaNs. conversely features whose median and means are further apart suggest the presence of outliers
and in this case I use ffill because we are dealing with sequentially ordered data and values in previous time steps are useful
in predicting values for later time steps'''
'''
import seaborn as sns
import matplotlib.pyplot as plt

# plot electricity data timestreams to see outliers
fig,ax = plt.subplots(3,sharex=True)
ax[0].plot(electricity_df['demand'], label='demand')
ax[1].plot(electricity_df['generation'], label='generation')
ax[2].plot(electricity_df['interchange'], label='interchange')
ax[0].set_title('Electricity data')
ax[0].set_ylabel('MWh')
ax[1].set_ylabel('MWh')
ax[2].set_ylabel('MWh')
ax[2].set_xlabel('Date')
plt.tight_layout()
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.savefig(WORKING_DIR + 'plots/electricity_data.png', dpi=350)
plt.close()

# plot histograms and violin plots as well as some stats for electricity data
for col in electricity_df.columns:
	fig,ax = plt.subplots(2, sharex=True)
	plot_data = electricity_df[col][~electricity_df[col].isnull()]
	ax[0].hist(plot_data, bins=15)
	title_text = r'$\sigma = %.2f$ MWh, $\mu = %.2f$ MWh, median $= %.2f$ MWh' % (plot_data.std(), plot_data.mean(), plot_data.median())
	sns.violinplot(plot_data, ax=ax[1])
	ax[0].set_ylabel('Count', labelpad=5)
	ax[1].set_xlabel('%s [MWh]' % col)
	ax[0].set_title(title_text, size=15)
	plt.tight_layout()
	plt.savefig(WORKING_DIR + 'plots/%s_dist.png' % col, dpi=350)
	plt.close()

# plot histograms and violin plots as well as some stats for weather data
for col in weather_df.columns:
	if col=='hourlyskyconditions': continue
	elif col=='hourlyprecip':
		title_text = r'$\sigma = %.2f$, $\mu = %.4f$, median $= %.4f$' % (plot_data.std(), plot_data.mean(), plot_data.median())
	else:
		title_text = r'$\sigma = %.2f$, $\mu = %.2f$, median $= %.2f$' % (plot_data.std(), plot_data.mean(), plot_data.median())
	plot_data = weather_df[col][~weather_df[col].isnull()]
	fig,ax = plt.subplots(2, sharex=True)
	ax[0].hist(plot_data, bins=15)
	sns.violinplot(plot_data, ax=ax[1])
	ax[0].set_ylabel('Count', labelpad=5)
	ax[1].set_xlabel('%s' % col)
	ax[0].set_title(title_text, size=15)
	plt.tight_layout()
	plt.savefig(WORKING_DIR + 'plots/%s_dist.png' % col, dpi=350)
	plt.close()

# plot bar plot for categorical value
weather_df['hourlyskyconditions'].value_counts().plot(kind='bar')
plt.close()
'''

# cut DFs based on date to align properly
cut_electricity = electricity_df[:'2018-09-01']
cut_weather = weather_df[cut_electricity.index.min():'2018-09-01']

# the date index columns for the weather and electricity data don't align exactly--see what's missing
elec_set = set(cut_electricity.index)
weather_set = set(cut_weather.index)
weather_set.difference(elec_set)

# based on the plots generated above, I choose certain columns to be filled with the median, others with ffill
fill_dict = {'median': ['dailyheatingdegreedays', 'generation', 'hourlyaltimetersetting', 'hourlydrybulbtempf', 'hourlyprecip', 'hourlysealevelpressure', 'hourlystationpressure', 'hourlywetbulbtempf', 'interchange', 'dailycoolingdegreedays', 'hourlyvisibility', 'hourlywindspeed', 'hourlycoolingdegrees', 'hourlyheatingdegrees'], 'ffill': ['demand', 'hourlydewpointtempf', 'hourlyrelativehumidity']}

# fill electricity data NaNs
for col in cut_electricity.columns:
	if col in fill_dict['median']:
		cut_electricity[col].fillna(cut_electricity[col].median(), inplace=True)
	else:
		cut_electricity[col].fillna(cut_electricity[col].ffill(), inplace=True)

# fill weather data NaNs
for col in cut_weather.columns:
	if col == 'hourlyskyconditions':
		cut_weather[col].fillna(cut_weather[col].value_counts().index[0], inplace=True) 
	elif col in fill_dict['median']:
		cut_weather[col].fillna(cut_weather[col].median(), inplace=True)
	else:
		cut_weather[col].fillna(cut_weather[col].ffill(), inplace=True)

# finally merge the data to get a complete dataframe for LA, ready for training
final_df = cut_weather.merge(cut_electricity, right_index=True, left_index=True, how='inner')

# save as pickle file
final_df.to_pickle(WORKING_DIR + 'LA_df.pkl')
