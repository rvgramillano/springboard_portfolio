import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'data/LA_df_first.pkl')
la_df = pd.get_dummies(la_df)
la_df = la_df.drop(['hourlyskyconditions_VV'], axis=1)

# plot electricity data timestreams to see outliers
fig,ax = plt.subplots()
ax.plot(la_df['demand'])
ax.set_title('Electricity data in LA', fontsize=16)
ax.set_ylabel('Electricity Demand [MWh]')
ax.set_xlabel('Date')
plt.tight_layout()
plt.savefig(WORKING_DIR + 'plots/distributions_%s/electricity_data.png' % 'Seattle', dpi=350)
plt.close()



# change depending on which city to analyze
df = la_df.copy()

# get pearson correlation coefficients for demand
print('DEMAND CORRELATIONS (PEARSON) FOR LA')
print(df.corr()['demand'].sort_values(ascending=False)[1:])

# get r^2 values per column and print
demand_r = {}
for col in df.columns:
	if col != 'demand':
		slope, intercept, r_value, p_value, std_err = scipy.stats.stats.linregress(df['demand'], df[col])
		demand_r[col] = r_value**2


print('DEMAND CORRELATIONS (r^2) FOR LA')
demand_r_df = pd.DataFrame({'col': demand_r.keys(), 'r^2': demand_r.values()})
print(demand_r_df.sort_values(by='r^2', ascending=False))

# here we store summary statistics of all the columns in a separate df for visualization
stds = df.std()
means = df.mean()
medians = df.median()

df_stats = pd.DataFrame({'std': stds, 'mean': means, 'median': medians})
print(df_stats.sort_values('std', ascending=False))

# below we generate collinearity plots for temperature and pressure

# pressure
for col in df.columns:
	if col not in ['hourlyaltimetersetting', 'hourlysealevelpressure', 'hourlystationpressure']:
		df = df.drop(col, axis=1)
from pandas.tools.plotting import scatter_matrix
axarr = scatter_matrix(df, alpha=.2)
ax = axarr[0,0]
labels = [item.get_text() for item in ax.get_yticklabels()]
ax.set_yticklabels([str(round(float(label), 2)) for label in labels])
plt.tight_layout()
plt.savefig(WORKING_DIR + 'plots/EDA/pressure_collin_la.png', dpi=300)
plt.close()

# temperature
df = la_df.copy()
for col in df.columns:
	if col not in ['hourlywetbulbtempf', 'hourlydrybulbtempf', 'hourlydewpointtempf']:
		df = df.drop(col, axis=1)
from pandas.tools.plotting import scatter_matrix
axarr = scatter_matrix(df, alpha=.2)
plt.tight_layout()
plt.savefig(WORKING_DIR + 'plots/EDA/temp_collin_la.png', dpi=300)
plt.close()

