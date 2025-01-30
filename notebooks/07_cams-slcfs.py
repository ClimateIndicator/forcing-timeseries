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

# %% [markdown]
# # Process CAMS SLCF emissions and compare them to CEDS
#
# CAMS includes agricultural waste burning in their sectoral totals, so need to remove this for comparison to CEDS.
#
# We see that there is quite a lot of disagreement between the datasets; let's use CAMS to extend CEDS by taking the ratio of CAMS in 2023 and 2024 to 2022

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

import warnings

# %%
species = ['SO2', 'BC', 'OC', 'NMVOC', 'NOx', 'NH3', 'CO']

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    emissions_annual = {}
    for specie in species:
        emissions_monthly = pd.read_csv(f'../data/slcf_emissions/cams/{specie}.csv', skiprows=10, index_col=0)
        emissions_monthly
        mi = pd.MultiIndex.from_tuples([(c[0:4], c[5:7]) for c in emissions_monthly.columns], names=['year', 'month'])
        emissions_monthly.columns = mi
        emissions_annual_all_sectors = emissions_monthly.groupby('year', axis=1).sum()
        emissions_annual[specie] = emissions_annual_all_sectors.loc['Sum_Sectors'] - emissions_annual_all_sectors.loc['Agricultural_waste_burning']

# %%
ceds_df = pd.DataFrame(columns = species, index=np.arange(2000, 2023, dtype=int))

for specie in species:
    ceds_df.loc[:, specie] = 0.001 * pd.read_csv(
        f'../data/slcf_emissions/ceds/v20240708/{specie}_CEDS_global_emissions_by_sector_v2024_07_08.csv'
    ).sum()['X2000':].values

ceds_df

# %%
fig, ax = pl.subplots(2, 4, figsize=(12, 6))
for ispec, specie in enumerate(species):
    irow = ispec // 4
    icol = ispec % 4
    ax[irow,icol].plot(np.arange(2000, 2025), emissions_annual[specie])
    ax[irow,icol].plot(np.arange(2000, 2023), ceds_df.loc[:, specie])
    ax[irow,icol].set_title(specie)

# %%
pd.DataFrame(emissions_annual).to_csv('../output/cams_2000-2024.csv')

# %%
