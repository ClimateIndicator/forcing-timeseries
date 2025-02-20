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
# Attempt to compare the datasets (page 21 of https://atmosphere.copernicus.eu/sites/default/files/publications/CAMS261_2021SC1_D6.1.2-2022_202306_Docu_v1_APPROVED_Ver1.pdf)
#
# Things in CAMS that are not in CEDS:
# - agricultural waste burning in their sectoral totals
# - the EDGAR sector "nfe" is part of industrial processes in CAMS, and according to the document is not included in CEDS. Because it is not separated out in CAMS, we can't isolate it from the total. However, nfe stands for non-ferrous metal production (https://essd.copernicus.org/preprints/essd-2023-306/essd-2023-306.pdf), which should correspond to CEDS sectors 2C3 and 2C4.
#
# Things in CEDS that are not in CAMS:
# - aviation; this is a separate dataset CAMS-GLOB-AIR. However, it doesn't seem that reliable, since there is no dip in emissions for COVID. The dataset also ends in 2022.
# - section 6 emissions ("Other"); but they seem to be zero for every species in CEDS
# - According to the comparison on p21 of the CAMS document, international shipping is not included in CAMS-GLOB-ANT. But looking at the 2019-2020 jump in SO2 and the shipping totals, I believe it is. So I'll treat CAMS as if it (now at least) includes international shipping
#
# **Therefore, my hypothesis is for the fairest, most reliable, most consistent comparison that we should compare total minus aviation in CEDS to total minus agricultural waste burning in CAMS.**
#
# From this comparison, we see that there is still quite a lot of disagreement between the datasets; but we use CAMS to extend CEDS by taking the ratio of CAMS in 2023 and 2024 to 2022.
#
# **CEDS 2023 should be ready in the not too distant future**
#
# I'm using the processed data from ScenarioMIP prepared by Marco Gambirini (the totals are basically identical to doing the following steps, but Marco has retained the sectoral detail). Marco adds aviation in, so I take it back out.
#
# Unfortunately there doesn't seem to be a way to automate downloads of CAMS data, so if you want to grab it:
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
cams_marco_raw_df = pd.read_csv('../data/slcf_emissions/cams/cams_world.csv')
cams_marco_raw_df

# %%
# Take out agricultural waste burning (in GFED, double counting) and aviation (not complete time series) from Marco's data
cams_marco_df = pd.DataFrame()
for specie in species:
    cams_marco_df[specie] = (
        cams_marco_raw_df[cams_marco_raw_df['variable'].str.contains(f'|{specie}|', regex=False)].loc[:, '2000':].sum() -
        cams_marco_raw_df[cams_marco_raw_df['variable']==f'Emissions|{specie}|Agricultural Waste Burning'].loc[:, '2000':].sum() -
        cams_marco_raw_df[cams_marco_raw_df['variable']==f'Emissions|{specie}|Aviation'].loc[:, '2000':].sum()
    )

# we know that cams NOx is NO units so put in NO2 units
cams_marco_df['NOx'] = cams_marco_df['NOx'] * 46.006 / 30.006
cams_marco_df

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

# we know that cams NOx is NO units so put in NO2 units
cams_df['NOx'] = cams_df['NOx'] * 46.006 / 30.006
cams_df

# %%
ceds_df = pd.DataFrame(columns = species, index=np.arange(2000, 2023, dtype=int))
ceds_no_aviation_df = pd.DataFrame(columns = species, index=np.arange(2000, 2023, dtype=int))

for specie in species:
    df_ceds_in = pd.read_csv(
        f'../data/slcf_emissions/ceds/v20240708/{specie}_CEDS_global_emissions_by_sector_v2024_07_08.csv'
    )
    total = df_ceds_in.sum()['X2000':].values
    aviation = df_ceds_in[df_ceds_in['sector'].isin(('1A3ai_International-aviation', '1A3aii_Domestic-aviation'))].sum()['X2000':].values
    ceds_df.loc[:, specie] = 0.001 * total
    ceds_no_aviation_df.loc[:, specie] = 0.001 * (total - aviation)

# %%
ceds_df

# %%
ceds_no_aviation_df

# %%
fig, ax = pl.subplots(2, 4, figsize=(12, 6))
for ispec, specie in enumerate(species):
    irow = ispec // 4
    icol = ispec % 4
#    ax[irow,icol].plot(np.arange(2000, 2026), cams_df.loc[:, specie], label='CAMS')
    ax[irow,icol].plot(np.arange(2000, 2026), cams_df.loc[:, specie], label="CAMS (excl. AIR & AWB)")
    # ax[irow,icol].plot(np.arange(2000, 2023), ceds_df.loc[:, specie], label='CEDS')
    ax[irow,icol].plot(np.arange(2000, 2023), ceds_no_aviation_df.loc[:, specie], label='CEDS (excl. AIR)')
    if specie in ['BC', 'OC']:
        ax[irow,icol].plot(np.arange(2000, 2026), 1.4 * cams_df.loc[:, specie], label='CAMS * 1.4')
    if specie in ['NH3']:
        ax[irow,icol].plot(np.arange(2000, 2026), 17/14 * cams_df.loc[:, specie], label='CAMS * 17/14')
    ax[irow,icol].set_title(specie)
    ax[irow,icol].set_ylim(0, 1.05 * np.max((cams_df.loc[:, specie].max(), ceds_df.loc[:, specie].max())))
    ax[irow,icol].legend();
pl.tight_layout()
pl.savefig('../plots/cams_ceds.png')
pl.savefig('../plots/cams_ceds.pdf')

# %%
cams_df.to_csv('../output/cams_2000-2025.csv')

# %%
