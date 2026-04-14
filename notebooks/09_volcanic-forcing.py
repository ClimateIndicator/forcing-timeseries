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
# # Volcanic forcing
#
# Use the AR6 process for volcanic forcing with updates:
# - replace Toohey & Sigl (eVolv v2) with Sigl et al. (HolVol) which extends back to 9500 BC. We use the full "pre-industrial" (9500 BC to 1749 AD) as the average background sAOD to reference zero forcing to.
# - use CMIP7 1750-2023
# - ues GloSSAC v2.2.3 until the end of 2024
# - Extend forwards to 2025 past the end of GlossAC using sAOD from OMPS LP, following the use of this dataset in https://www.nature.com/articles/s43247-022-00580-w#Sec11
#
# This notebook requires large datasets that need registration to obtain so cannot be downloaded by the code:
#
# - Download data from: https://download.pangaea.de/dataset/928646/files/HolVol_SAOD_-9500_1900_v1.0.nc Put this in '../data/volcanic_aod/HolVol_SAOD_-9500_1900_v1.0.nc'
# - Download data from: https://asdc.larc.nasa.gov/project/GloSSAC/GloSSAC_2.23. Click "Get Dataset". Put this in '../data/volcanic_aod/GloSSAC_V2.23.nc'
# - Download data from: https://omps.gesdisc.eosdis.nasa.gov/data/SNPP_OMPS_Level3/OMPS_NPP_LP_L3_AER_MONTHLY.1.1/. Obtain every *.h5 file in every annual sub-directory from 2013 to 2025. Put these files in '../data/volcanic_aod/SNPP_OMPS_Level3/ . **Note: following both the non-updating of the v1.0 retrieval and the personal communication from Ghassan Taha, we now use v1.1**
#
# In addition, we require estimates of the stratospheric water vapour injection from Hunga Tonga. The MLS data is processed in notebook 09, taken and run in the offline radiative transfer code, and implemented back here.
#
# For bonus points we should consider stratospheric water vapour in historical eruptions, though this is not easy to get.

# %%
import glob

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from pooch import retrieve
import scipy.stats
import h5py

# %%
df_stwv = pd.read_csv('../data/volcanic_stwv/indicators_stwv_irf_hansen_tp.csv', index_col=0)

# %%
pl.rcParams['figure.figsize'] = (10/2.54, 10/2.54)
pl.rcParams['font.size'] = 11
pl.rcParams['font.family'] = 'Arial'
pl.rcParams['ytick.direction'] = 'in'
pl.rcParams['ytick.minor.visible'] = True
pl.rcParams['ytick.major.right'] = True
pl.rcParams['ytick.right'] = True
pl.rcParams['xtick.direction'] = 'in'
pl.rcParams['xtick.minor.visible'] = True
pl.rcParams['xtick.major.top'] = True
pl.rcParams['ytick.major.left'] = True
pl.rcParams['xtick.top'] = True
pl.rcParams['figure.dpi'] = 150
pl.rcParams['axes.spines.top'] = True
pl.rcParams['axes.spines.bottom'] = True

# %%
# -9500 to 1900
nc = Dataset('../data/volcanic_aod/HolVol_SAOD_-9500_1900_v1.0.nc')
aod550_evolv = nc.variables['aod550'][:]
lat_evolv = nc.variables['lat'][:]
time_evolv = nc.variables['time'][:]
nc.close()
time_evolv[-51*12]

# %%
lat_evolv_bnds = np.concatenate([[90], 0.5*(lat_evolv[1:]+lat_evolv[:-1]), [-90]])
weights = -np.squeeze(np.diff(np.sin(np.radians(lat_evolv_bnds))))
aod_evolv = np.zeros((len(time_evolv)))
for i in range(len(time_evolv)):
    aod_evolv[i] = np.average(aod550_evolv[i,:],weights=weights)

# %%
# 1979 to 2024 from GloSSAC
nc = Dataset('../data/volcanic_aod/GloSSAC_V2.23_NC4.nc')
data_glossac = nc.variables['Glossac_Aerosol_Optical_Depth'][:]
lat_glossac = nc.variables['lat'][:]
trp_hgt_glossac = nc.variables['trp_hgt'][:]  # lat, month
alt_glossac = nc.variables['alt'][:]
nc.close()
data_glossac[0,0,:]

# %%
lat_glossac_bnds = np.concatenate(([-90], 0.5*(lat_glossac[1:]+lat_glossac[:-1]), [90]))
weights_glossac = np.diff(np.sin(np.radians(lat_glossac_bnds)))

# Glossac is at 525 nm. 2.33 Angstrom exponent from Kovilakam et al 2020, https://essd.copernicus.org/articles/12/2607/2020/essd-12-2607-2020.html
angstrom = (550/525)**(-2.33)

months_glossac = 12*(2025-1979)
aod_glossac = np.zeros(months_glossac)
for i in range(months_glossac):
    aod_glossac[i] = np.average(data_glossac[i,:,2],weights=weights_glossac)*angstrom

# %%
# 1750 to 2023 from CMIP7
# cmip7_file = retrieve(
#     'https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/CMIP/uoexeter/UOEXETER-CMIP-1-3-0/atmos/mon/ext/gnz/v20250127/ext_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-1-3-0_gnz_175001-202312.nc',
#     known_hash='d95814c1859a6d5a96bb986e6e4395dfeed2aedd18d256de5ba8f425356c8187',
#     progressbar=True
# )
# cmip6_file = '../data/volcanic_aod/CMIP_1850_2014_extinction_550nm_strat_only_v3.nc'
cmip7_file = "../data/volcanic_aod/ext_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_175001-202312.nc"

# if the download was working, you'd get it from here:
#'https://esgf-data2.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP7/CMIP/uoexeter/UOEXETER-CMIP-2-2-1/atmos/mon/ext/gnz/v20250521/ext_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_175001-202312.nc',

# %%
nc = Dataset(cmip7_file)

# %%
nc.variables['wavelength'][7]

# %%
nc.variables

# %%
nc.variables['ext'][:, :, :, 7]

# %%
ext_cmip7 = nc.variables['ext'][:, :, :, 7].transpose((0,2,1)) * 1000  # time, height, lat; convert per m to per km
lev_cmip7 = nc.variables['height'][:]
lat_cmip7 = nc.variables['lat'][:]
time_cmip7 = nc.variables['time'][:]
nc.close()

# %%
time_cmip7

# %%
lev_cmip7.shape

# %%
lat_cmip7_bnds = np.concatenate(([-90], 0.5*(lat_cmip7[1:]+lat_cmip7[:-1]), [90]))
weights = np.diff(np.sin(np.radians(lat_cmip7_bnds)))
months_cmip7 = 12*(2024-1750)
tax = np.zeros(months_cmip7)
aod_cmip7 = np.zeros(months_cmip7)

for i in range(months_cmip7):
    aod_cmip7[i] = np.average(np.nansum(ext_cmip7[i,...] * 0.5, axis=0), weights=weights)

# %%
pl.plot(aod_cmip7)
np.sum(np.isnan(aod_cmip7))

# %%
months_omps = 12 * (2026-2013)
aod_omps = np.zeros(months_omps)
aod_omps_unscaled = np.zeros(months_omps)

# %%
# 745 nm band is index 3, and is the most reliable (at least in v1.0).
# 1.9 Angstrom exponent from Taha et al 2021, https://amt.copernicus.org/articles/14/1015/2021/
angstrom = (550/745)**(-1.9)

for i in range(months_omps):
    year = (i)//12 + 2013
    month = ((i-12)%12)+1
    # missing dataset
    # if year==2024 and month==7:
    #     continue
    filename = glob.glob('../data/volcanic_aod/SNPP_OMPS_Level3/OMPS-NPP_LP-L3-AER-MONTHLY-ym_v1.1_%4dm%02d_*.h5' % (year, month))[0]
    h5 = h5py.File(filename)
    lat_omps = h5['/']['Latitude'][:]
    lat_omps_bnds = np.concatenate(([-90], 0.5*(lat_omps[1:]+lat_omps[:-1]), [90]))
    weights = np.diff(np.sin(np.radians(lat_omps_bnds)))
    data = h5['/']['StratColumn'][:]
    data[data==-999] = np.nan
    aod_omps_unscaled[i] = np.nansum(data[:,:,3] * weights * np.ones((24, 36)))/((weights * np.ones((24, 36)))[~np.isnan(data[:,:,3])].sum())

# # fill in july 2024 as an average of june and august
# aod_omps_unscaled[-18] = np.mean((aod_omps_unscaled[-19], aod_omps_unscaled[-17]))

# %%
#aod_omps = aod_glossac[-108:].mean()/aod_omps_unscaled[:108].mean() * aod_omps_unscaled
# calculate scaling factor between OMPS and GloSSAC for the overlap period 2013-2022
#aod_omps = aod_glossac[-(months_omps-12):].mean()/aod_omps_unscaled[:(months_omps-12)].mean() * aod_omps_unscaled
aod_omps = aod_omps_unscaled * angstrom

# %%
aod_glossac[:12]

# %%
# eVolv -9500 to 1749 = 135 000 months
# CMIP7 1750 to 1978 = 2748 months
# GloSSAC 1979 to 2012 = 408 months
# OMPS 2013 to 2025 = 156 months
# sum = 138 312 months
#aod = np.concatenate((aod_evolv[:135000], aod_cmip7, aod_glossac[-12:], aod_omps[-12:]))
aod = np.concatenate((aod_evolv[:135000], aod_cmip7[:2748], aod_glossac[:408], aod_omps[:]))
len(aod)

# %%
aod_omps_unscaled

# %%
pl.plot(np.arange(2013+1/24, 2024, 1/12), aod_cmip7[-132:], label='CMIP7 (550 nm)')
pl.plot(np.arange(2013+1/24, 2025, 1/12), aod_glossac[-(months_omps-12):], label='GloSSAC v2.2.3 (550 nm)')
pl.plot(np.arange(2013+1/24, 2026, 1/12), aod_omps_unscaled, label='OMPS LP (600 nm)')
pl.plot(np.arange(2013+1/24, 2026, 1/12), aod_omps, label='OMPS LP scaled')
pl.ylabel('sAOD')
pl.legend(fontsize=8)
#pl.plot(aod[-120:])
pl.savefig('../plots/AOD_comparison.png')
pl.savefig('../plots/AOD_comparison.pdf')

# %%
## crossfade
#aod[135000:135612] = (1-np.linspace(0,1,612))*aod_evolv[135000:135612]+np.linspace(0,1,612)*aod_cmip7[:612]  # 51 years holvol to cmip7
#aod[138276:138288] = (1-np.linspace(0,1,12))*aod_cmip7[-12:] + np.linspace(0, 1, 12)*aod_omps[-24:-12] # 1 year cmip7 to omps

# %%
pl.plot(np.arange(1745+1/24,1801+1/24,1/12), aod_evolv[134940:135612], label='HolVol')
pl.plot(np.arange(1750+1/24,1805+1/24,1/12), aod_cmip7[:660], label='CMIP7')
pl.plot(np.arange(1745+1/24,1805+1/24,1/12), aod[134940:135660], label='blended')
pl.legend()

# %%
pl.plot(np.arange(1979+1/24,2025+1/24,1/12), aod_glossac, label='GloSSAC v2.2.3 (1979-2024)', lw=0.5)
pl.plot(np.arange(1975+1/24,2024+1/24,1/12), aod_cmip7[-588:], label='CMIP6 (1850-2023)', lw=0.5)
pl.plot(np.arange(2013+1/24,2026+1/24,1/12), aod_omps, label='OMPS (2013-2025)', lw=0.5)
pl.plot(np.arange(1975+1/24,2026+1/24,1/12), aod[-612:], label='Combined', zorder=-2, color='k')
pl.title('Stratospheric aerosol optical depth')
pl.ylabel("AOD")
pl.xlim(1975, 2026)
pl.ylim(0, 0.12)
pl.tight_layout()
pl.legend(frameon=False, fontsize=8)
pl.tight_layout()
pl.savefig('../plots/volcanic_AOD.png')
pl.savefig('../plots/volcanic_AOD.pdf')

# %%
erf_sulf = -20 * (aod - np.mean(aod[:(11262*12)]))
erf_h2o = np.zeros_like(erf_sulf)

for im, month in enumerate(range(216, 264)):
    mod12 = month%12       
    erf_h2o[-48+im] = (df_stwv.iloc[month]-df_stwv.iloc[mod12:18*12:12].mean())


# fill January 2025 as a mean of December 2024 and February 2025, as it is missing
erf_h2o[-12] = 0.5 * (erf_h2o[-13] + erf_h2o[-11])

erf = erf_sulf + erf_h2o

# %%
erf_h2o[-48:-36].mean()

# %%
erf_h2o[-36:-24].mean()

# %%
erf_h2o[-24:-12].mean()

# %%
erf_h2o[-12:].mean()

# %%
pl.plot(np.arange(1975+1/24,2026+1/24,1/12), erf_sulf[-(12*51):], label='AOD', color='skyblue')
pl.plot(np.arange(2021+23/24,2026,1/12), erf_h2o[-49:], label='H$_2$O', color='green')
pl.plot(np.arange(2021+23/24,2026,1/12), erf[-49:], label='AOD + H$_2$O', color='purple')
pl.title('Volcanic effective radiative forcing')
pl.xlim(1975, 2026)
pl.ylim(-2.1, 0.4)
pl.ylabel('W m$^{-2}$, relative to 9500 BCE to 1749 CE')
pl.axhline(0, ls=':', color='k')
pl.legend(frameon=False)
pl.tight_layout()
pl.savefig('../plots/volcanic_ERF.png')
pl.savefig('../plots/volcanic_ERF.pdf')

# %%
months = np.arange(-9500+1/24,2026,1/12)
df = pd.DataFrame(
    data=np.vstack([aod, erf_sulf, erf_h2o, erf]).T, 
    index=months, 
    columns=['stratospheric_AOD', 'ERF_sulfate', 'ERF_H2O', 'volcanic_ERF']
)
df.index.name = 'year'
df.to_csv('../output/volcanic_sAOD_ERF_monthly.csv')

# %%
years = np.arange(-9500 + 0.5, 2026)
aod_ann = np.zeros(len(aod)//12)
sul_ann = np.zeros(len(erf)//12)
h2o_ann = np.zeros(len(erf)//12)
erf_ann = np.zeros(len(erf)//12)
for i in range(0, len(months), 12):
    aod_ann[i//12] = np.mean(aod[i:i+12])
    sul_ann[i//12] = np.mean(erf_sulf[i:i+12])
    h2o_ann[i//12] = np.mean(erf_h2o[i:i+12])
    erf_ann[i//12] = np.mean(erf[i:i+12])
df = pd.DataFrame(data=np.vstack([aod_ann, sul_ann, h2o_ann, erf_ann]).T, index=years, columns=['stratospheric_AOD', 'ERF_sulfate', 'ERF_H2O', 'volcanic_ERF'])
df.index.name = 'year'
df.to_csv('../output/volcanic_sAOD_ERF_annual.csv')

# %%
df

# %%
pl.plot(df.volcanic_ERF)

# %%
