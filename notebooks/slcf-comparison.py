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
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

import os

# %%
pl.rcParams['figure.figsize'] = (14/2.54, 22/2.54)
pl.rcParams['font.size'] = 7
pl.rcParams['font.family'] = 'Arial'
pl.rcParams['xtick.direction'] = 'out'
pl.rcParams['xtick.minor.visible'] = True
#pl.rcParams['ytick.major.left'] = True
pl.rcParams['ytick.direction'] = 'out'
pl.rcParams['ytick.minor.visible'] = True
#pl.rcParams['ytick.major.size'] = 0
pl.rcParams['xtick.top'] = True
pl.rcParams['ytick.right'] = True
pl.rcParams['figure.dpi'] = 150

# %%
indicators2023 = pd.read_csv('../output/slcf_emissions_1750-2023.csv', index_col=0)

# %%
indicators2022 = pd.read_csv('../data/slcf_emissions/indicators_2022/slcf_emissions_1750-2022.csv', index_col=0)

# %%
#cams_shipping

ceds_so2_df = pd.read_csv(
    f'../data/slcf_emissions/ceds/v20240401/SO2_CEDS_global_emissions_by_sector_v2024_04_01.csv'
)
ceds_so2_df.loc[ceds_so2_df['sector']=='1A3di_International-shipping'].sum()['X2000':].values / 1000
cams_df = pd.read_csv('../data/slcf_emissions/cams/cams-glob-ship-anthro-so.csv', index_col=0)
#cams_df[:, 'Total'] = 
cams_df['Total'] = cams_df.loc[:, 'SO2'] + 96.06/64.066 * cams_df.loc[:, 'SO4']

#cams_shipping = indicators2023.loc[2000:2023, 'SO2']

cams_shipping = indicators2023.loc[2000:2022, 'SO2'] + cams_df['Total'] - ceds_so2_df.loc[ceds_so2_df['sector']=='1A3di_International-shipping'].sum()['X2000':].values / 1000
cams_shipping

# %%
fig, ax = pl.subplots(4,2)

for ispec, specie in enumerate(indicators2023.columns):
    row = ispec//2
    col = ispec%2
    ax[row, col].plot(np.arange(1950.5, 2023), indicators2022.loc[1950:, specie], label='Indicators 2022')
    ax[row, col].plot(np.arange(1950.5, 2024), indicators2023.loc[1950:, specie], label='Indicators 2023')
    if specie=='SO2':
        ax[row, col].plot(np.arange(2000.5, 2023), cams_shipping.loc[2000:], label='Indicators 2023 + CAMS-GLOB-SHIP', zorder=-7)
        ax[row, col].legend()
    ax[row, col].set_xlim(1950, 2025)
    ax[row, col].set_ylim(0, ax[row, col].get_ylim()[1])
    ax[row, col].set_ylabel('Tg / yr')
    ax[row, col].set_title(specie)
ax[3,1].axis('off')
fig.tight_layout()

os.makedirs('../plots/', exist_ok=True)
pl.savefig('../plots/slcf-comparison.png')
pl.savefig('../plots/slcf-comparison.pdf')

# %%
