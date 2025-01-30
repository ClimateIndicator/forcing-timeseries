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
# # Prepare and understand MLS stratospheric water vapour
#
# We will use this as estimate for Hunga Tonga positive forcing.
#
# To obtain the data, go to https://search.earthdata.nasa.gov/search?q=MLS%20H2O&fi=MLS&fl=3%2B-%2BGridded%2BObservations. Select the MLS/Aura Level 3 Monthly Binned Water Vapor (H2O) Mixing Ratio on Assorted Grids V005 (ML3MBH2O). Download this data to ../data/mls.
#
# **update 29.01.2025: 2024 data not yet ready (and may not ever be, with the retirement of the instrument)**
#
# Perhaps we should just use ERA5?

# %%
from netCDF4 import Dataset
import matplotlib.pyplot as pl
import numpy as np
import glob
from fair.earth_params import mass_atmosphere
import xarray as xr

# %%
data = np.ones((20*12, 39)) * np.nan

# %%
for year in range(2004, 2024):
    nc = Dataset(glob.glob(f'../data/mls/MLS-Aura_L3MB-H2O_*_{year}.nc')[0])
    data[(year-2004)*12:(year-2004)*12+12, :] = nc.groups['H2O PressureZM'].variables['value'][:, 10:49, 15:30].mean(axis=2)
    plev = nc.groups['H2O PressureZM'].variables['lev'][10:49]
    nc.close()

# %%
data

# %%
X, Y = np.meshgrid(np.arange(2004+1/24, 2024, 1/12), plev)

# %%
data[data==0] = np.nan

# %%
pl.contourf(
    np.arange(2004+1/24, 2024, 1/12),
    plev[:],
    data.T
)
ax = pl.gca()
ax.set_ylim(ax.get_ylim()[::-1])
pl.colorbar()

# %%
plev

# %%
plev_bounds = 0.5 * (plev[1:] + plev[:-1])
plev_bounds = np.append(plev_bounds, [0])

# %%
plev_bounds

# %%
era5_plev = np.array([125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1])
era5_plev_bounds = 0.5 * (era5_plev[1:] + era5_plev[:-1])
era5_plev_bounds = np.append(era5_plev_bounds, [0])
era5_plev_bounds

# %%
# weighting fractions
(112.5-plev_bounds[1]) / (plev_bounds[0] - plev_bounds[1])  # 10 into 100 hPa
(85 - plev_bounds[3]) / (plev_bounds[2] - plev_bounds[3])  # 12 into 70 hPa (rest into 100 hPa)
(60 - plev_bounds[5]) / (plev_bounds[4] - plev_bounds[5])  # 14 into 50 hPa (rest into 70 hPa)
(40 - plev_bounds[7]) / (plev_bounds[6] - plev_bounds[7])  # 16 into 30 hPa (rest into 50 hPa)
(25 - plev_bounds[9]) / (plev_bounds[8] - plev_bounds[9])  # 18 into 20 hPa (rest into 30 hPa)
(15 - plev_bounds[12]) / (plev_bounds[11] - plev_bounds[12])  # 21 into 10 hPa (rest into 20 hPa)
(8.5 - plev_bounds[15]) / (plev_bounds[14] - plev_bounds[15])  # 24 into 7 hPa (rest into 10 hPa)
(6 - plev_bounds[17]) / (plev_bounds[16] - plev_bounds[17])  # 26 into 5 hPa (rest into 7 hPa)
(4 - plev_bounds[19]) / (plev_bounds[18] - plev_bounds[19])  # 28 into 3 hPa (rest into 5 hPa)
(2.5 - plev_bounds[21]) / (plev_bounds[20] - plev_bounds[21])  # 30 into 2 hPa (rest into 3 hPa)
(1.5 - plev_bounds[24]) / (plev_bounds[23] - plev_bounds[24])  # 33 into 1 hPa (rest into 2 hPa)

# %%
weighting_fractions = np.zeros((38, 11))
weighting_fractions[0, 0] = (112.5-plev_bounds[1]) / (plev_bounds[0] - plev_bounds[1])
weighting_fractions[1, 0] = 1
weighting_fractions[2, 0] = 1 - (85 - plev_bounds[3]) / (plev_bounds[2] - plev_bounds[3])
weighting_fractions[2, 1] = (85 - plev_bounds[3]) / (plev_bounds[2] - plev_bounds[3])
weighting_fractions[3, 1] = 1
weighting_fractions[4, 1] = 1 - (60 - plev_bounds[5]) / (plev_bounds[4] - plev_bounds[5])
weighting_fractions[4, 2] = (60 - plev_bounds[5]) / (plev_bounds[4] - plev_bounds[5])  # 14 into 50 hPa (rest into 70 hPa)
weighting_fractions[5, 2] = 1
weighting_fractions[6, 2] = 1 - (40 - plev_bounds[7]) / (plev_bounds[6] - plev_bounds[7]) 
weighting_fractions[6, 3] = (40 - plev_bounds[7]) / (plev_bounds[6] - plev_bounds[7])  # 16 into 30 hPa (rest into 50 hPa)
weighting_fractions[7, 3] = 1
weighting_fractions[8, 3] = 1 - (25 - plev_bounds[9]) / (plev_bounds[8] - plev_bounds[9])
weighting_fractions[8, 4] = (25 - plev_bounds[9]) / (plev_bounds[8] - plev_bounds[9])  # 18 into 20 hPa (rest into 30 hPa)
weighting_fractions[9, 4] = 1
weighting_fractions[10, 4] = 1
weighting_fractions[11, 4] = 1 - (15 - plev_bounds[12]) / (plev_bounds[11] - plev_bounds[12])
weighting_fractions[11, 5] = (15 - plev_bounds[12]) / (plev_bounds[11] - plev_bounds[12])  # 21 into 10 hPa (rest into 20 hPa)
weighting_fractions[12, 5] = 1
weighting_fractions[13, 5] = 1
weighting_fractions[14, 5] = 1 - (8.5 - plev_bounds[15]) / (plev_bounds[14] - plev_bounds[15])
weighting_fractions[14, 6] = (8.5 - plev_bounds[15]) / (plev_bounds[14] - plev_bounds[15])  # 24 into 7 hPa (rest into 10 hPa)
weighting_fractions[15, 6] = 1
weighting_fractions[16, 6] = 1 - (6 - plev_bounds[17]) / (plev_bounds[16] - plev_bounds[17])
weighting_fractions[16, 7] = (6 - plev_bounds[17]) / (plev_bounds[16] - plev_bounds[17])  # 26 into 5 hPa (rest into 7 hPa)
weighting_fractions[17, 7] = 1
weighting_fractions[18, 7] = 1 - (4 - plev_bounds[19]) / (plev_bounds[18] - plev_bounds[19])
weighting_fractions[18, 8] = (4 - plev_bounds[19]) / (plev_bounds[18] - plev_bounds[19])  # 28 into 3 hPa (rest into 5 hPa)
weighting_fractions[19, 8] = 1
weighting_fractions[20, 8] = 1 - (2.5 - plev_bounds[21]) / (plev_bounds[20] - plev_bounds[21])
weighting_fractions[20, 9] = (2.5 - plev_bounds[21]) / (plev_bounds[20] - plev_bounds[21])  # 30 into 2 hPa (rest into 3 hPa)
weighting_fractions[21, 9] = 1
weighting_fractions[22, 9] = 1
weighting_fractions[23, 9] = 1 - (1.5 - plev_bounds[24]) / (plev_bounds[23] - plev_bounds[24])
weighting_fractions[23, 10] = (1.5 - plev_bounds[24]) / (plev_bounds[23] - plev_bounds[24])  # 33 into 1 hPa (rest into 2 hPa)
weighting_fractions[24:, 10] = 1


import pandas as pd
pd.DataFrame(weighting_fractions, index=np.arange(11, 49), columns = era5_plev[1:])

# %% [markdown]
# ## Need to conservatively regrid this to ERA5's plev / lon / lat
#
# - Start from VMR in MLS, convert to MMR (specific humidity)
# - Calculate actual mass of water vapour in each cell on MLS pressure levels
# - Use the lookup table of mapping MLS pressure levels to ERA5 pressure levels to calculate mass of water in each ERA5 cell
# - Convert back to specific humidity

# %%
data = np.ma.masked_all((20*12, 38, 72, 45)) * np.nan
data

# %%
for year in range(2004, 2024):
    nc = Dataset(glob.glob(f'../data/mls/MLS-Aura_L3MB-H2O_*_{year}.nc')[0])
    data[(year-2004)*12:(year-2004)*12+12, ...] = nc.groups['H2O PressureGrid'].variables['value'][:, 11:49, ...]
    plev = nc.groups['H2O PressureGrid'].variables['lev'][10:49]
    lat = nc.groups['H2O PressureGrid'].variables['lat'][:]
    lon = nc.groups['H2O PressureGrid'].variables['lon'][:]
    lat_bnds = nc.groups['H2O PressureGrid'].variables['lat_bnds'][:]
    lon_bnds = nc.groups['H2O PressureGrid'].variables['lon_bnds'][:]
    nc.close()

# %%
era5_data = np.ma.masked_all((20*12, 11, 72, 45))
era5_h2o_mass = np.ma.masked_all((20*12, 11, 72, 45))

# %%
mass_atmosphere # slice

# %%
p_top = 0
p_bottom = 112.5
mass_slice = (p_bottom-p_top)/1000 * mass_atmosphere

# %%
era5_plevthick = -np.diff(era5_plev_bounds)
era5_plevthick

# %%
# era5_mass = era5_plevthick * mass_atmosphere # how much does the stratospheric slice weigh in kg

# %%
plev

# %%
plev_bounds = 0.5 * (plev[:-1] + plev[1:])
plev_bounds = np.append(plev_bounds, [0])
plev_bounds

# %%
plev_diff = -np.diff(plev)
plev_thick = 0.5 * (plev_diff[1:] + plev_diff[:-1])
plev_thick = np.append(plev_thick, 0.5 * (plev[-2] + plev[-1]))
plev_thick

# %%
lon_lat_weight = np.diff(np.sin(np.radians(lat_bnds))).squeeze() / 2 / 72

# %%
# mass of water vapour at each cell
mass_h2o = (plev_thick[None, :, None, None] / 1000 * mass_atmosphere * lon_lat_weight * data) * 18.015 / 28.97
# the last ratio is VMR to MMR
mass_h2o.shape

# %%
mass_h2o.sum(axis=(1,2,3))  # in kg
mass_h2o.sum(axis=(1,2,3))/1e9  # in Tg

# %%
pl.plot(mass_h2o.sum(axis=(1,2,3))/1e9)

# %%
# climatology
mass_h2o_20042021 = np.ma.masked_all((12))
for month in range(12):
    mass_h2o_20042021[month] = np.nanmean(mass_h2o[month:216:12].sum(axis=(1,2,3))/1e9)

# %%
mass_h2o_2022 = np.ma.masked_all(12)
for month in range(12):
    mass_h2o_2022[month] = mass_h2o[month+216].sum()/1e9

# %%
mass_h2o_2023 = np.ma.masked_all(12)
for month in range(12):
    mass_h2o_2023[month] = mass_h2o[month+228].sum()/1e9

# %%
mass_h2o_2022

# %%
mass_h2o_2023

# %%
(mass_h2o_2022 - mass_h2o_20042021)

# %%
(mass_h2o_2023 - mass_h2o_20042021)

# %%
weighting_fractions.shape

# %%
era5_plev

# %%
for ilev in range(11):
    era5_h2o_mass[:, ilev, ...] = np.sum(mass_h2o * weighting_fractions[:, ilev][None, :, None, None], axis=1)

# %%
# Then I need to convert this back to a mass mixing ratio for socrates

# %%
pl.plot(era5_h2o_mass.sum(axis=(1,2,3))/1e9)

# %%
# climatology
era5_h2o_mass_20042021 = np.ma.masked_all((12))
for month in range(12):
    era5_h2o_mass_20042021[month] = np.nanmean(era5_h2o_mass[month:216:12].sum(axis=(1,2,3))/1e9)

era5_h2o_mass_2022 = np.ma.masked_all(12)
for month in range(12):
    era5_h2o_mass_2022[month] = era5_h2o_mass[month+216].sum()/1e9

era5_h2o_mass_2023 = np.ma.masked_all(12)
for month in range(12):
    era5_h2o_mass_2023[month] = era5_h2o_mass[month+228].sum()/1e9

# %%
era5_h2o_mass_2022 - era5_h2o_mass_20042021

# %%
era5_h2o_mass_2023 - era5_h2o_mass_20042021

# %%
era5_h2o_mass

# %%
# convert back to MMR in each cell

era5_data = 1000 * era5_h2o_mass / era5_plevthick[None, :, None, None] / (mass_atmosphere * lon_lat_weight)
#mass_h2o = (plev_thick[None, :, None, None] / 1000 * mass_atmosphere * lon_lat_weight * data) * 18.015 / 28.97

era5_data

# %%
X, Y = np.meshgrid(np.arange(2004+1/24, 2024, 1/12), era5_plev[1:])

# %%
pl.contourf(
    np.arange(2004+1/24, 2024, 1/12),
    era5_plev[1:],
    era5_data[..., 15:30].mean(axis=(2,3)).T
)
ax = pl.gca()
ax.set_ylim(ax.get_ylim()[::-1])
pl.colorbar()

# %%
# climatology
era5_data_20042021 = np.ma.masked_all((12, 11, 72, 45))
for month in range(12):
    era5_data_20042021[month, ...] = np.nanmean(era5_data[month:216:12, ...], axis=0)

era5_data_2022 = np.ma.masked_all((12, 11, 72, 45))
for month in range(12):
    era5_data_2022[month, ...] = era5_data[month+216, ...]

era5_data_2023 = np.ma.masked_all((12, 11, 72, 45))
for month in range(12):
    era5_data_2023[month, ...] = era5_data[month+228, ...]

# %%
era5_ds = xr.Dataset(
    data_vars = dict(
        h2o_mmr = (["time", "plev", "lat", "lon"], era5_data.transpose(0,1,3,2))
    ),
    coords=dict(
        lon=lon,
        lat=lat,
        time=np.arange(240),
        plev = era5_plev[1:]
    ),
)

# %%
era5_ds.to_netcdf('../output/MLS_H2O.nc')

# %%
lon

# %%
lat

# %%
