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
#
# **CEDS 2023 should be ready in the not too distant future**
#
# **Lara Aleluia (CMCC) will check the CAMS emissions as part of ScenarioMIP - good to cross ref**
#
# Unfortunately there doesn't seem to be a way to automate downloads of CAMS data. Do this:
#
# - https://eccad.sedoo.fr/#/data
# - select CAMS-GLOB_ANT v6.2
# - select species
# - select Sum Sectors and Agricultural Waste Burning

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

import warnings

# %%
species = ['SO2', 'BC', 'OC', 'NMVOC', 'NOx', 'NH3', 'CO']

# %%
cams_df = pd.DataFrame()
species_subs = {specie: specie for specie in species}
species_subs['NMVOC'] = 'NMV'
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    emissions_annual = {}
    for specie in species:
        emissions_monthly = pd.read_csv(f'../data/slcf_emissions/cams/cams-glob-ant-anthro-{species_subs[specie].lower()}.csv', parse_dates=['Date'])

        cams_df[specie] = (
            emissions_monthly.groupby(emissions_monthly.Date.dt.year)[' Sum Sectors'].sum() -
            emissions_monthly.groupby(emissions_monthly.Date.dt.year)[' Agricultural waste burning'].sum()
        )
cams_df

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
    ax[irow,icol].plot(np.arange(2000, 2026), cams_df.loc[:, specie], label='CAMS')
    ax[irow,icol].plot(np.arange(2000, 2023), ceds_df.loc[:, specie], label='CEDS')
    if specie=='OC':
        ax[irow,icol].plot(np.arange(2000, 2026), 1.4 * cams_df.loc[:, specie], label='CAMS * 1.4')
    elif specie=='NOx':
        ax[irow,icol].plot(np.arange(2000, 2026), 46.006/30.006 * cams_df.loc[:, specie], label='CAMS * 1.53')
    ax[irow,icol].set_title(specie)
    ax[irow,icol].set_ylim(0, 1.05 * np.max((cams_df.loc[:, specie].max(), ceds_df.loc[:, specie].max())))
    ax[irow,icol].legend();
pl.tight_layout()
pl.savefig('../plots/cams_ceds.png')
pl.savefig('../plots/cams_ceds.pdf')

# %%
cams_df.to_csv('../output/cams_2000-2025.csv')

# %%
