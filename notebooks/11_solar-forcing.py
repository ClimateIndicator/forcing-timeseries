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
# # Update IPCC AR6 ERF timeseries
#
# - PMIP4
# - CMIP7 from EGSF
# - TSI-CDR: https://www.ncei.noaa.gov/data/total-solar-irradiance/access/

# %%
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# %% [markdown]
# ### Solar radiation
#
# No change from AR6 - yet - so not re-generated here - using the data provided as-is.
#
# net change in TOA radiation = delta TSI * 1/4 * 0.71 * 0.72 where 1/4 is geometric factor, 0.71 is planetary co-albedo, 0.72 is rapid adjustment factor. Pre-industrial is defined as the mean TSI for complete solar cycles from 6755 BCE to 1750 CE.
#
# On top of this we add a linear trend from 1750 to 2019 based on the PD uncertainty assessment of +/- 0.07 W m-2.

# %%
solar_df = pd.DataFrame(index=np.arange(-6755,2026))
solar_df.index.name = 'Year'

# %%
nc = Dataset('../data/solar/pmip4/SSI_14C_cycle_yearly_cmip_v20160613_fc.nc')
wl_bin     = nc.variables['wavelength_bin'][:]
time_pmip4 = nc.variables['time'][:]
ssi_pmip4  = nc.variables['ssi'][:]
nc.close()

# %%
years = np.arange(1850, 2016, dtype=int)
steps = np.ones(166, dtype=int) * 365
steps[np.logical_and(years%4==0, np.logical_or(years%100!=0, years%400==0))] = 366
idx_1850 = np.argmin(time_pmip4<1850)
idx_yearend = idx_1850+np.cumsum(steps)
idx_yearstart = np.insert(idx_yearend, 0, [idx_1850])[:-1] 
# print (idx_yearstart[:10], idx_yearend[:10])
# years_future = np.arange(2016, 2300)
# isleap = np.zeros(284)
# isleap[np.logical_and(years_future%4==0, np.logical_or(years_future%100!=0, years_future%400==0))] = 1

# %%
ssi_pmip4.shape

# %%
idx_yearstart

# %%
solar_df.loc[-6755:1849, 'pmip4'] = np.sum(ssi_pmip4[:idx_1850, :]*wl_bin, axis=1)

# %%
year = 1850
for i, idx in enumerate(idx_yearstart):
    solar_df.loc[year, 'pmip4'] = np.sum(np.mean(ssi_pmip4[idx_yearstart[i]:idx_yearend[i], :], axis=0)*wl_bin)
    year = year+1

# %%
solar_df.loc[1845:1855]

# %%
nc = Dataset('../data/solar/cmip7/multiple_input4MIPs_solar_CMIP_SOLARIS-HEPPA-CMIP-4-6_gn_185001-202312.nc')
cmip7_tsi = nc.variables['tsi'][:]
cmip7_monthlengths = (nc.variables["time_bnds"][:, 1] - nc.variables["time_bnds"][:, 0])
#monthlengths = (cmip6.time_bnds[:, 1] - cmip6.time_bnds[:, 0]).astype('timedelta64[D]') / np.timedelta64(1, 'D')
nc.close()

# %%
cmip7_tsi_annual = np.ones(cmip7_tsi.shape[0]//12) * np.nan
for year in np.arange(len(cmip7_tsi_annual)):
    cmip7_tsi_annual[year] = np.sum(cmip7_tsi[year*12:year*12+12] * cmip7_monthlengths[year*12:year*12+12]) / np.sum(cmip7_monthlengths[year*12:year*12+12])

# %%
solar_df.loc[1850:2023, 'cmip7'] = cmip7_tsi_annual

# %%
solar_df

# %%
# get annual data from here too!

#nc = Dataset('../data/solar/tsi_cdr/tsi_v03r00_monthly_s187501_e187512_c20240831.nc')
nc = Dataset('../data/solar/tsi_cdr/tsi_v03r00_yearly_s1610_e2025_c20260305.nc')
cdr_tsi_yearly = nc.variables['TSI'][:]
nc.close()

# %%
solar_df.loc[1610:2025, 'cdr_tsi'] = cdr_tsi_yearly

# %%
solar_df

# %%
pl.plot(solar_df)
pl.xlim(1610, 2026)
#pl.legend()

# %%
# minima 1644, 1657, 1671, 1685, 1698
solar_df.loc[1642:1701, 'pmip4']

# %%
# minima 1644, 1657, 1671, 1685, 1698
solar_df.loc[[1644, 1657, 1671, 1685, 1698], 'cdr_tsi'] - solar_df.loc[[1644, 1657, 1671, 1685, 1698], 'pmip4']

# %%
solar_df.loc[1642:1701, 'cdr_tsi'] - solar_df.loc[1642:1701, 'pmip4']

# %%
pl.plot(solar_df['cdr_tsi'])
pl.plot(solar_df['pmip4']+0.787109)
pl.xlim(1610, 1857)

# %%
solar_df.loc[:, 'pmip4_rebaselined'] = solar_df.loc[:, 'pmip4'] + 0.787109

# %%
pl.plot(solar_df['pmip4_rebaselined'])
pl.plot(solar_df['cdr_tsi'])
pl.xlim(1643, 1660)

# %%
solar_df.loc[-6755:1643, 'igcc'] = solar_df.loc[-6755:1643, 'pmip4_rebaselined']
solar_df.loc[1644:2025, 'igcc'] = solar_df.loc[1644:2025, 'cdr_tsi']

# %%
pl.plot(solar_df['igcc'])

# %%
solar_df.loc[-6755:-6720, 'igcc']  #6754
solar_df.loc[1740:1755, 'igcc'] #1743
solar_df.loc[:, 'erf'] = 0.25 * 0.71 * 0.72 * (solar_df['igcc'] - solar_df.loc[-6754:1743, 'igcc'].mean())

# %%
pl.plot(solar_df['erf'])

# %%
solar_df.to_csv('../output/solar_tsi_erf.csv')

# %%
