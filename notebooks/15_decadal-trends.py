# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# Read in the CSV file with the ERF data
erf_df = pd.read_csv('../output/ERF_best_aggregates_1750-2024.csv')

# %%
erf_df = erf_df.rename(columns={'Unnamed: 0': 'Year'})

# %%
# Set the 'Year' column as the index
erf_df.set_index('Year', inplace=True)

# %%
rolling_mean = erf_df.rolling(window=10).mean()

# %%
erf_trend = (rolling_mean - rolling_mean.shift(10))

# %%
y=erf_trend['anthro'].values
x=erf_trend.index.values

# %%
# Read in the CSV file with the temperature data
GWI_df = pd.read_csv('../data/gmst/Walsh_GMST_timeseries.csv', index_col=0)
GWI_df

# %%
rolling_mean_GWI = GWI_df.rolling(window=10).mean()
GWI_trend = (rolling_mean_GWI - rolling_mean_GWI.shift(10))
y_GWI=GWI_trend['anthropogenic_p50'].values
x_GWI=GWI_trend.index.values+0.5

# %%
# Create a figure and axis object
fig, ax1 = plt.subplots()

# Create a twin axis object
ax2 = ax1.twinx()

# Plot each series on its respective axis
ax2.plot(x,y,marker='o', linestyle='None',label='effective radiative forcing trend')
ax2.plot(x[-5:], y[-5:], marker='o', color='red',linestyle='None')

ax1.plot(x_GWI,y_GWI,marker='+', linestyle='None',label='human-induced warming trend')
ax1.plot(x_GWI[-4:], y_GWI[-4:], marker='+', color='red',linestyle='None')

# Add axis labels and legend
ax1.set_xlabel('End year of decade')
ax2.set_ylabel('ERF trend (Wm$^{-2}$decade$^{-1}$)')
ax1.set_ylabel('Temperature trend ( °C decade$^{-1}$)')

ax1.legend(loc='best')
ax2.legend(loc='center right')
plt.title('Decadal trends')
plt.xlim([1970,2025])

# Show the plot
plt.show()
fig.savefig('../plots/decadal_trends.png')
fig.savefig('../plots/decadal_trends.pdf')

# %%
