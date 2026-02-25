# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Download and process biomass burning emissions
#
# New for 2025: The GFED data is now on an FTP server and can't be directly downloaded over HTTP. Therefore, use an SFTP client and log in thus:
#
# https://www.globalfiredata.org/ancill/GFED5_SFTP_info.txt
#
# We will continue to use GFED4.1s while this is still updated.

# %%
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as pl

# %%
hashes = {
    'emissions_factors': '5f68c5c4ffdb7d81d3d2fefa662dcad9dd66f2b4097350a08a045523626383b2',
}

files = {}

for year in range(1997, 2017):
    files[year] = f"../data/slcf_emissions/gfed/GFED4.1s_{year}.hdf5"

for year in range(2017, 2026):
    files[year] = f"../data/slcf_emissions/gfed/GFED4.1s_{year}_beta.hdf5"

files['emissions_factors'] = pooch.retrieve(
    "https://www.geo.vu.nl/~gwerf/GFED/GFED4/ancill/GFED4_Emission_Factors.txt",
    hashes['emissions_factors']
)

efs = pd.read_csv(files['emissions_factors'], comment='#', sep=r'\s+', index_col=0, header=None)
efs.columns = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']
efs.index.rename('SPECIE', inplace=True)


efs.loc['MEK', 'SAVA']
sources=list(efs.columns)
species=list(efs.index)

months       = '01','02','03','04','05','06','07','08','09','10','11','12'
start_year = 1997
end_year   = 2025


"""
make table with summed DM emissions for each region, year, and source
"""
table = np.zeros((41, end_year - start_year + 1)) # region, year

for year in range(start_year, end_year+1):
    print(year)
    f = h5py.File(files[year], 'r')


    if year == start_year: # these are time invariable
        grid_area     = f['/ancill/grid_cell_area'][:]

    emissions = np.zeros((41, 720, 1440))
    for month in range(12):
        # read in DM emissions
        string = '/emissions/'+months[month]+'/DM'
        DM_emissions = f[string][:]
        for ispec, specie in enumerate(species):
            for isrc, source in enumerate(sources):
                # read in the fractional contribution of each source
                string = '/emissions/'+months[month]+'/partitioning/DM_'+source
                contribution = f[string][:]
                # calculate emissions as the product of DM emissions (kg DM per
                # m2 per month), the fraction the specific source contributes to
                # this (unitless), and the emission factor (g per kg DM burned)
                emissions[ispec, ...] += DM_emissions * contribution * efs.loc[specie, source]
                #print(emissions[:, 88, 0])


    # fill table with total values for the globe (row 15) or basisregion (1-14)
    #mask = np.ones((720, 1440))
    table[:, year-start_year] = np.sum(grid_area[None, ...] * emissions, axis=(1,2))

table = table / 1E12

# %%
species=list(efs.index)
gfed41s_df = pd.DataFrame(table.T, index=range(start_year, end_year+1), columns=species)
gfed41s_df['NMVOC'] = gfed41s_df.loc[:,'C2H6':'C3H6O'].sum(axis=1) + gfed41s_df.loc[:,'C2H6S':].sum(axis=1)

# %%
gfed41s_df

# %%
os.makedirs('../output/', exist_ok=True)
gfed41s_df.to_csv('../output/gfed4.1s_1997-2024.csv')

# %%
