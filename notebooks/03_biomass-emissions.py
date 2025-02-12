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
# # Download and process biomass burning emissions
#
# **note: as of 2025-02-12, the 2024 beta does not yet exist**
#
# Also, the hashes of the 2017-2022 "beta" files have changed on 2023-09-18, which means the data may also have changed.
#
# Make an extrapolation based on last five year average

# %%
import os
import numpy as np
import pandas as pd
import pooch
import h5py
import matplotlib.pyplot as pl

# %%
hashes = {
 'emissions_factors': '5f68c5c4ffdb7d81d3d2fefa662dcad9dd66f2b4097350a08a045523626383b2',
 1997: '997f54a532cae524757c3b35808c10ae0f71ce231c213617cb34ba4b72968bb9',
 1998: '36c13cdcec4f4698f3ab9f05bc83d2307252d89b81da5a14efd8e171148a6dc0',
 1999: '5d0d18b09d9a76e305522c5b46a97bf3180d9301d1d3c6bfa5a4c838fb0fa452',
 2000: 'ddbeff2326dded0e2248afd85c3ec7c84a36c6919711632e717d00985cd4ad6d',
 2001: '1b684bf0b348e92a5d63ea660564f01439f69c4eb88eacd46280237d51ce5815',
 2002: 'dcf624961512dbb93759248bc2b75d404b3be68f1f6fdcb01f0c7dc7f11a517a',
 2003: '91d61b67d04b4a32d534f5d68ae1de7929f7ea75bb9d25d3273c4d5d75bda4d3',
 2004: '931e063f796bf1f7d391d3f03342d2dd2ad1b234cb317f826adfab201003f4cd',
 2005: '159e7704d14089496d051546c20b644a443308eeb7d79bf338226af2b4bdc2b7',
 2006: 'a69d5bf6b8fa3324c2922aac07306ec6e488a850ca4f42d09a397cee30eebd4c',
 2007: '1d7f77e6f7b13cc2a8ef9d26ecb9ea3d18e70cfeb8a47e7ecb26f9613888f937',
 2008: 'bd3771b9b3032d459a79c0da449fdb497cd3400e0e07a0da6b41e930fc5d3e14',
 2009: '36ea9b6036cd0ff3672502c3c04180bd209ddb192f86a2e791a2b896308bc5ff',
 2010: '5b2d30b5ddc3e20c38c7971faf6791b313b1bbff22e8bc2b14ca7ea9079aa12c',
 2011: 'fb19c001bef26ca23d07dd8978fd998f4692bdecdec5eb86b91d4b1ffb4a9aa7',
 2012: '08033c90295bbc208fac426e01809b68cef62997668085b1e096d8a61ab43e9b',
 2013: 'cf5249811af4b7099f886e61125dcd15c1127b6125392fe8358d3f0bf8ddb064',
 2014: 'a293b4c6e03898a0dc184a082a37435673916a02ff02c06668152dcc4d4b8405',
 2015: 'c043e96a421247afbeb6580fca0bcddf8160180b14d37b13122fc3110534b309',
 2016: '2f3b54ff5698ba7f7aa2bb1d4b5e5f95124c0e0db32830ed94aa04bea2cbc2a6',
 2017: 'a9859da022e97853efd1ce89664f31d1e8c0cddac2f35472d1e445d019f2a927',
 2018: '4c08e36c2d7b1bc7b1020a15d36421bdddc68a4cc08ee9f1069d23b49f3cf34b',
 2019: '8ac86c5d35e7ddfe9dbe6d71982ea54f6ee3b5a43d9ebbef19ef2957992587e6',
 2020: '83b0ba1f5080cd2d19265c05e0c66f9563ad085381541f8c75d31e69ae839b99',
 2021: '5bf68b48515b04fe0dbb493c9b2b6e564a5b206da4af499ef5492e4d835516c5',
 2022: '46d1e90287c0c012eb1dbb8b7aa2d0c90bfb617b5c708da3c5a8ffa26ca22abb',
 2023: 'a9c90c311080e3bd2671322369e38d71a478a518777abbea17d2430fb73424ea'
}

files = {}
for year in range(1997, 2017):
    files[year] = pooch.retrieve(
        f"https://www.geo.vu.nl/~gwerf/GFED/GFED4/GFED4.1s_{year}.hdf5",
        f"{hashes[year]}"
    )
for year in range(2017, 2024):
    files[year] = pooch.retrieve(
        f"https://www.geo.vu.nl/~gwerf/GFED/GFED4/GFED4.1s_{year}_beta.hdf5",
        None
    )

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

# in this example we will calculate annual CO emissions for the 14 GFED
# basisregions over 1997-2014. Please adjust the code to calculate emissions
# for your own specie, region, and time period of interest. Please
# first download the GFED4.1s files and the GFED4_Emission_Factors.txt
# to your computer and adjust the directory where you placed them below


# we are interested in CO for this example (4th row):
start_year = 1997
end_year   = 2023


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
gfed41s_df.loc[2024, :] = gfed41s_df.loc[2019:2023, :].mean(axis=0)

# %%
gfed41s_df

# %%
os.makedirs('../output/', exist_ok=True)
gfed41s_df.to_csv('../output/gfed4.1s_1997-2024.csv')

# %%
