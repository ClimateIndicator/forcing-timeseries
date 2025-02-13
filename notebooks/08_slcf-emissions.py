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
# # SLCF emissions
#
# Fossil and industrial from CEDS (1750-2022), from CEDS v2024.07.08, available at https://doi.org/10.5281/zenodo.12803197
#
# Extend with CAMS for 2023 and 2024 based on CAMS ratio to 2022 emissions
#
# Biomass burning from GFED (1997-2023), extended backwards to 1750 using BB4CMIP **TODO: replace with the CMIP7 pipeline**, which is part of the CMIP6 database and taken here from RCMIP. We need to convert the unit of NOx emissions from biomass to NO2 from the RCMIP data, as GFED reports in units of NO.

# %%
import os
import numpy as np
import pandas as pd
import pooch
import h5py
import matplotlib.pyplot as pl

# %%
species = ['BC', 'OC', 'SO2', 'NOx', 'CO', 'NMVOC', 'NH3']

# %%
slcf_df = pd.DataFrame(columns = species, index=np.arange(1750, 2025, dtype=int))
ceds_df = pd.DataFrame(columns = species, index=np.arange(1750, 2023, dtype=int))

# %%
for specie in species:
    ceds_df.loc[:, specie] = 0.001 * pd.read_csv(
        f'../data/slcf_emissions/ceds/v20240708/{specie}_CEDS_global_emissions_by_sector_v2024_07_08.csv'
    ).sum()['X1750':].values

# %%
ceds_df

# %%
gfed41s_df = pd.read_csv('../output/gfed4.1s_1997-2024.csv', index_col=0)

# %%
gfed41s_df

# %%
cams_df = pd.read_csv('../output/cams_2000-2025.csv', index_col=0)
cams_df

# %%
rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

# RCMIP
rcmip_in_df = pd.read_csv(rcmip_emissions_file)

# %%
gfed_convert = {specie: 1 for specie in species}
gfed_convert['NOx'] = 46.006/30.006  # molecular weight ratio of NO2 to NO
rcmip_specie = {specie: specie for specie in species}
rcmip_specie['NMVOC'] = 'VOC'
rcmip_specie['SO2'] = 'Sulfur'

# %%
cams_df.loc[2023:] / cams_df.loc[2022] * ceds_df.loc[2022]

# %%
gfed41s_df.loc[2023:2024, specie] * gfed_convert[specie]

# %%
for specie in species:
    rcmip_df = rcmip_in_df.loc[
        (rcmip_in_df['Scenario']=='historical')&
        (rcmip_in_df['Region']=='World')&
        (rcmip_in_df['Variable'].str.startswith(f'Emissions|{rcmip_specie[specie]}|')),
    :]
    #ceds_rcmip = [f'Emissions|{rcmip_specie}|MAGICC AFOLU|Agriculture', f'Emissions|{rcmip_specie}|MAGICC Fossil and Industrial']
    uva_rcmip = [
        f'Emissions|{rcmip_specie[specie]}|MAGICC AFOLU|Agricultural Waste Burning',
        f'Emissions|{rcmip_specie[specie]}|MAGICC AFOLU|Forest Burning',
        f'Emissions|{rcmip_specie[specie]}|MAGICC AFOLU|Grassland Burning',
        f'Emissions|{rcmip_specie[specie]}|MAGICC AFOLU|Peat Burning'
    ]
    
    
    slcf_df.loc[1750:1996, specie] = (
        ceds_df.loc[1750:1996, specie] + (
            rcmip_df.loc[rcmip_df['Variable'].isin(uva_rcmip), '1750':'1996']
            .interpolate(axis=1)
            .sum()
            .values.
            squeeze() * gfed_convert[specie]
        )
    )

    slcf_df.loc[1997:2022, specie] = (
        ceds_df.loc[1997:2022, specie] +
        gfed41s_df.loc[1997:2022, specie].values.squeeze() * gfed_convert[specie]
    )

    # assume ratio for 2023 and 2024 in CEDS based on 2022 in CAMS
    slcf_df.loc[2023:2024, specie] = (
        cams_df.loc[2023:2024, specie] / cams_df.loc[2022, specie] * ceds_df.loc[2022, specie] + 
        gfed41s_df.loc[2023:2024, specie] * gfed_convert[specie]
    )

# %%
fig, ax = pl.subplots(2, 4, figsize=(12, 6))
for ispec, specie in enumerate(species):
    irow = ispec // 4
    icol = ispec % 4
    ax[irow,icol].plot(slcf_df.loc[2000:2024, specie])
    ax[irow,icol].set_title(specie)

# %%
slcf_df

# %%
os.makedirs('../output', exist_ok=True)
slcf_df.to_csv('../output/slcf_emissions_1750-2024.csv')

# %%
