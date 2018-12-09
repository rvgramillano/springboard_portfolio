import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, LSTM

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

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'data/LA_df.pkl')

dataset = la_df.copy()

#set the column we want to predict (demand) to the first columns for consistency
cols = list(dataset.columns)
cols.remove('demand')
cols.insert(0,'demand')
dataset = dataset[cols]

#{col:index+1 for col,index in zip(dataset.columns,range(len(dataset.columns)))}
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, dataset.columns, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]], axis=1, inplace=True)

'''The first step is to prepare the pollution dataset for the LSTM.

This involves framing the dataset as a supervised learning problem and normalizing the input variables.

We will frame the supervised learning problem as predicting the electricity demand at the current hour (t) given the electricity demand and weather conditions at the prior time step.

This formulation is straightforward and just for this demonstration. Some alternate formulations you could explore include:

Predict the pollution for the next hour based on the weather conditions and pollution over the last 24 hours.
Predict the pollution for the next hour as above and given the “expected” weather conditions for the next hour.

This data preparation is simple and there is more we could explore. Some ideas you could look at include:

One-hot encoding wind speed.
Making all series stationary with differencing and seasonal adjustment.
Providing more than 1 hour of input time steps.
This last point is perhaps the most important given the use of Backpropagation through time by LSTMs when learning sequence prediction problems.
'''
'''
First, we must split the prepared dataset into train and test sets. To speed up the training of the model for this demonstration, we will only fit the model on the first year of data, then evaluate it on the remaining data. 
'''
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


'''
We will define the LSTM with 50 neurons in the first hidden layer and 1 neuron in the output layer for predicting pollution. The input shape will be 1 time step with 17 features.

We will use the Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent.

The model will be fit for 50 training epochs with a batch size of 72. Remember that the internal state of the LSTM in Keras is reset at the end of each batch, so an internal state that is a function of a number of days may be helpful (try testing this).

Finally, we keep track of both the training and test loss during training by setting the validation_data argument in the fit() function. At the end of the run both the training and test loss are plotted.
'''



# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.savefig(WORKING_DIR + 'plots/modeling/lstm_train_test.png', dpi=300)
plt.show()


'''After the model is fit, we can forecast for the entire test dataset.

We combine the forecast with the test dataset and invert the scaling. We also invert scaling on the test dataset with the expected pollution numbers.

With forecasts and actual values in their original scale, we can then calculate an error score for the model. In this case, we calculate the Root Mean Squared Error (RMSE) that gives error in the same units as the variable itself.
'''
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
rsq = r2_score(inv_y, inv_yhat)
print('Test RMSE: %.5f' % rmse)
print('Test r^2: %.5f' % rsq)





EIA_API = '70b5193b6dcb775ee0a0d947bc60f55a'

def EIA_request_to_df(req, value_name):
	'''
	This function unpacks the JSON file into a pandas dataframe.'''
	dat = req['series'][0]['data']
	dates = []
	values = []
	for date, value in dat:
		if value is None:
			continue
		dates.append(date)
		values.append(float(value))
	df = pd.DataFrame({'date': dates, value_name: values})
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.sort_index()
	return df


# collect electricty data for Los Angeles
REGION_CODE = 'LDWP'

# megawatthours
url_demand_forecast = requests.get('http://api.eia.gov/series/?api_key=%s&series_id=EBA.%s-ALL.DF.H' % (EIA_API, REGION_CODE)).json()
electricity_df = EIA_request_to_df(url_demand_forecast, 'demand_forecast')

# cut demand forecast in the same way we did for demand
cut_electricity = electricity_df[:'2018-09-01']
elec_i = dataset[['demand']]

# join demand forecast with demand to align dataframes
elec_join = elec_i.join(cut_electricity, how='left')
# delete first entry; this is what was done in the beginning to frame the problem as supervised (features at t-1 and features at t requires deleting the first element)
elec_join = elec_join.iloc[1:]
# cut demand forecast just as we did for testing data set 
electricity_compare = elec_join[['demand_forecast']].values[n_train_hours:, :]
# find indices where no value was recorded for demand forecast
nan_inds = np.where(np.isnan(electricity_compare)==True)[0]
# print how many nan values we have for demand forecast--it's ~.15%
nan_percent = len(nan_inds) / float(len(electricity_compare))
print('percent missing from demand forecast: %.5f' % (nan_percent*100))
# get non-nan inds for cutting
non_nan_inds = np.where(np.isnan(electricity_compare)!=True)[0]
# remove nan values from demand forecast
electricity_compare_cut = electricity_compare[non_nan_inds].flatten()
# remove those same values from the actual demand to keep consistent
inv_y_cut = inv_y[non_nan_inds]
# compute rmse
forecast_rmse = np.sqrt(mean_squared_error(inv_y_cut, electricity_compare_cut))
forecast_rsq = r2_score(inv_y_cut, electricity_compare_cut)

print('Forecast RMSE: %.3f' % forecast_rmse)
print('Forecast r^2: %.3f' % forecast_rsq)






