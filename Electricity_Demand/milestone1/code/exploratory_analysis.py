import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

WORKING_DIR = '/Users/rvg/Documents/springboard_ds/springboard_portfolio/Electricity_Demand/'

la_df = pd.read_pickle(WORKING_DIR + 'data/LA_df.pkl')
seattle_df = pd.read_pickle(WORKING_DIR + 'data/seattle_df.pkl')

# plot electricity data timestreams to see outliers
fig,ax = plt.subplots()
ax.plot(seattle_df['demand'])
ax.set_title('Electricity data in Seattle', fontsize=16)
ax.set_ylabel('Electricity Demand [MWh]')
ax.set_xlabel('Date')
plt.tight_layout()
plt.savefig(WORKING_DIR + 'plots/distributions_%s/electricity_data.png' % 'Seattle', dpi=350)
plt.close()


# The following code creates scatterplots of the weather features (x-variable) to the electricity demand feature to examine relationships
for col in la_df.columns:
	if col=='hourlyskyconditions' or col=='demand': 
		continue
	print col
	fig,ax = plt.subplots(1,2)
	sns.regplot(x=col, y='demand', data=la_df, ax=ax[0], scatter_kws={'alpha':0.1})
	sns.regplot(x=col, y='demand', data=seattle_df, ax=ax[1], scatter_kws={'alpha':0.1})
	ax[0].set_title('LA correlation')
	ax[1].set_title('Seattle correlation')
	ax[0].set_ylabel('Demand [MWh]')
	ax[1].set_ylabel('Demand [MWh]')
	plt.tight_layout()
	plt.savefig(WORKING_DIR + 'plots/scatter_comparisons/%s.png' % col, dpi=300)
	#plt.show()
	#raw_input('...')
	plt.close()


# change depending on which city to analyze
df = la_df.copy()

# get pearson correlation coefficients for demand
print('DEMAND CORRELATIONS (PEARSON) FOR SEATTLE')
print(df.corr()['demand'].sort_values(ascending=False)[1:])

# get r^2 values per column and print
demand_r = {}
for col in df.columns:
	if col == 'hourlyskyconditions': continue
	if col != 'demand':
		slope, intercept, r_value, p_value, std_err = scipy.stats.stats.linregress(df['demand'], df[col])
		demand_r[col] = r_value**2


print('DEMAND CORRELATIONS (r^2) FOR SEATTLE')
demand_r_df = pd.DataFrame({'col': demand_r.keys(), 'r^2': demand_r.values()})
print(demand_r_df.sort_values(by='r^2', ascending=False))

# here we store summary statistics of all the columns in a separate df for visualization
stds = df.std()
means = df.mean()
medians = df.median()

df_stats = pd.DataFrame({'std': stds, 'mean': means, 'median': medians})
print(df_stats.sort_values('std', ascending=False))

# below we perform hypothesis testing with the HDD and CDD features to determine the significance of their relationships with electrictiy demand
demand_p = {}
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)   
    
    return corr_mat[0,1]

for col in ['dailycoolingdegreedays', 'dailyheatingdegreedays', 'hourlyheatingdegrees', 'hourlycoolingdegrees']:
	#if col=='demand' or col == 'hourlyskyconditions': continue
	print col
	# y-variable
	y = np.array(df['demand'])
	
	# x-variable
	x = np.array(df[col])
	
	# Compute observed correlation: r_obs
	r_obs = pearson_r(x, y)
	
	# Initialize permutation replicates: perm_replicates
	perm_replicates = np.empty(10000)
	print r_obs
	# Draw replicates
	for i in range(10000):
	    # Permute illiteracy measurments: illiteracy_permuted
	    x_permuted = np.random.permutation(x)
	
	    # Compute Pearson correlation
	    perm_replicates[i] = pearson_r(x_permuted, y)
	
	# Compute p-value: p
	if r_obs > 0:
		p = np.sum(perm_replicates >= r_obs) / float(len(perm_replicates))
	elif r_obs < 0:
		p = np.sum(perm_replicates <= r_obs) / float(len(perm_replicates))
	demand_p[col] = p
	print 'p-value = %.8f' % p

