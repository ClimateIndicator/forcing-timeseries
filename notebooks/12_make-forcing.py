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
# # Update IPCC AR6 ERF timeseries
#
# - with updated **preliminary** data to 2024
#
# **NOTE**: we continue to use CMIP6 solar forcing, because
#
# 1. CMIP7 does not go back before 1850;
# 2. CMIP7 historical ends in 2023, and no future dataset is yet prescribed (see https://solarisheppa.geomar.de/cmip7)
#
# 2024 differences compared to 2023:
# - using CAMS to extend CEDS beyond 2022
# - water vapour from HTHH is an estimate for 2024
# - GFED is assumed mean of last 5 years beyond 2023
# - stratospheric water vapour scales with methane concentrations not methane forcing
# - land use change is from LUH2 + GCB2024 for albedo and FAO for irrigation, replacing the cumulative CO2 AFOLU estimate **and we also update the uncertainties to take into account the separate assessments on these ranges**

# %%
import copy
import json

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scipy.stats
from tqdm.auto import tqdm
from fair.forcing.ghg import meinshausen2020
import xarray as xr

# %%
# probablistic ensemble
SAMPLES = 100000
forcing = {}
forcing_ensemble = {}

# %%
NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
NINETY_TO_ONESIGMA

# %%
# Required adjustment to each species to ensure overall halogenated gas ERF uncertainty is around 19%
HALOGEN_SCALING = 2.05

# %%
with open('../data/random_seeds.json', 'r') as filehandle:
    SEEDS = json.load(filehandle)

# %%
emissions = pd.read_csv('../output/slcf_emissions_1750-2024.csv', index_col=0)
emissions

# %%
concentrations = pd.read_csv('../output/ghg_concentrations_1750-2024.csv', index_col=0)
for year in range(1751, 1850):
    concentrations.loc[year, :] = np.nan
concentrations.sort_index(inplace=True)
concentrations.interpolate(inplace=True)

# %%
concentrations

# %%
# uncertainties from IPCC
uncertainty_seed = 38572

unc_ranges = {
    'CO2':          0.12/NINETY_TO_ONESIGMA,
    'CH4':          0.20/NINETY_TO_ONESIGMA,
    'N2O':          0.14/NINETY_TO_ONESIGMA,
    'HFC-125':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-134a':     0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-143a':     0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-152a':     0.26/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-227ea':    0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-23':       0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-236fa':    0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-245fa':    0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-32':       0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-365mfc':   0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HFC-43-10mee': 0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'NF3':          0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'C2F6':         0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'C3F8':         0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'n-C4F10':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'n-C5F12':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'n-C6F14':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'i-C6F14':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'C7F16':        0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'C8F18':        0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CF4':          0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'c-C4F8':       0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'SF6':          0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'SO2F2':        0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CCl4':         0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-11':       0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-112':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-112a':     0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-113':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-113a':     0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-114':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-114a':     0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-115':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-12':       0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CFC-13':       0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CH2Cl2':       0.26/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CH3Br':        0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CH3CCl3':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CH3Cl':        0.26/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'CHCl3':        0.26/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HCFC-124':     0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HCFC-133a':    0.26/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HCFC-141b':    0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HCFC-142b':    0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HCFC-22':      0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'HCFC-31':      0.26/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'Halon-1211':   0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'Halon-1301':   0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'Halon-2402':   0.19/NINETY_TO_ONESIGMA*HALOGEN_SCALING,
    'O3':           0.50/NINETY_TO_ONESIGMA,      # Total ozone
    'H2O_stratospheric': 1.00/NINETY_TO_ONESIGMA,      # stratospheric WV from CH4
    'contrails':    0.70/NINETY_TO_ONESIGMA,      # contrails approx - half-normal
    'BC_on_snow':   1.25/NINETY_TO_ONESIGMA,      # bc on snow - half-normal
    'land_albedo':  (2/3)/NINETY_TO_ONESIGMA,      # land use change  -0.25 to -0.05 W/m2 ONE STANDARD DEVIATION: GHIMIRE (IPCC chapter might be a typo as says ± 0.01 W/m2)
    'volcanic':     0.25/NINETY_TO_ONESIGMA,  # volcanic
    'solar': 0.5/NINETY_TO_ONESIGMA,      # solar (amplitude)
}

# %%
scale_df = pd.DataFrame(
    scipy.stats.norm.rvs(
        size=(SAMPLES, 59), 
        loc=np.ones((SAMPLES, 59)), 
        scale=pd.Series(unc_ranges), 
        random_state=uncertainty_seed
    ), columns=unc_ranges.keys()
)


# %%
def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


# %%
lapsi_params = scipy.optimize.root(opt, [1, 1, 1], args=(0, 1, 2.25)).x
irrigation_params = scipy.optimize.root(opt, [1.5, 0.5, 1], args=(-1, 1, 2)).x  # -0.1 to +0.05, best -0.05, so -1 to +2 x best
contrails_params = scipy.optimize.root(opt, [1, 1, 1], args=(19 / 57, 1, 98 / 57)).x

# %%
## skewnormal for asymmetric distributions
scale_df['BC_on_snow'] = scipy.stats.skewnorm.rvs(
    lapsi_params[0],
    loc=lapsi_params[1],
    scale=lapsi_params[2],
    size=SAMPLES,
    random_state=3701584,
)

scale_df['contrails'] = scipy.stats.skewnorm.rvs(
    contrails_params[0],
    loc=contrails_params[1],
    scale=contrails_params[2],
    size=SAMPLES,
    random_state=3701585,
)

scale_df['irrigation'] = scipy.stats.skewnorm.rvs(
    irrigation_params[0],
    loc=irrigation_params[1],
    scale=irrigation_params[2],
    size=SAMPLES,
    random_state=13710426,
)

trend_solar = scipy.stats.norm.rvs(
    size=SAMPLES, 
    loc=0.00, 
    scale=0.07/NINETY_TO_ONESIGMA, 
    random_state=uncertainty_seed
)

# %%
irrigation_params

# %%
pl.hist(scale_df['irrigation'], bins=50)

# %% [markdown]
# ### Solar radiation
#
# No change from AR6 - yet - so not re-generated here - using the data provided as-is.
#
# net change in TOA radiation = delta TSI * 1/4 * 0.71 * 0.72 where 1/4 is geometric factor, 0.71 is planetary co-albedo, 0.72 is rapid adjustment factor. Pre-industrial is defined as the mean TSI for complete solar cycles from 6755 BCE to 1750 CE.
#
# On top of this we add a linear trend from 1750 to 2019 based on the PD uncertainty assessment of +/- 0.07 W m-2.

# %%
df_solar = pd.read_csv('../data/ar6/solar_erf.csv', index_col=0)
forcing['solar'] = df_solar.loc[1750:2024].values.squeeze()

# %%
df_solar.loc[2019:2024]

# %%
df_solar.loc[2009:2019].mean()  # cycle 24 was Dec 2008 to Dec 2019: https://en.wikipedia.org/wiki/Solar_cycle_24

# %%
# as AR6, trend extended to 2024, but with 2019 end
trend = np.ones(275)
trend[:270] = np.linspace(0, 1, 270)

# %%
forcing_ensemble['solar'] = trend[:,None] * trend_solar[None,:] + forcing['solar'][:,None] * scale_df['solar'].values[None,:]

# %% [markdown]
# ### Volcanic forcing
#
# Calculated in a previous notebook

# %%
df_volcanic = pd.read_csv('../output/volcanic_sAOD_ERF_annual_-9500-2024.csv', index_col=0)
forcing['volcanic'] = df_volcanic.loc[1750.5:2024.5, 'volcanic_ERF'].values.squeeze()

# %% [markdown]
# ### Aerosol forcing
#
# In AR6, ERFari was based on emissions to forcing coefficients from Myhre et al (2013) https://acp.copernicus.org/articles/13/1853/2013/. At the time, I deemed there not sufficient evidence from CMIP6 AerChemMIP models or any other sources to update these. The uncertainty ranges from each precursor were expanded slightly compared to Myhre et al., in order to reproduce the overall ERFari uncertainty assessment (assumed that uncertainties in individual components are uncorrelated).
#
# Following AR6 and a re-calibration of FaIR, I now use Bill Collins/Terje Bertnsen/Sara Blichner/Sophie Szopa's chapter 6 correspondences of emissions or concentrations to forcing.
#
# ERFaci is based on fits to CMIP6 models from Smith et al. (2021) now updated to include 13 models and correct APRP code from Mark Zelinka.
#
# Rescale both to the assessed forcings of -0.3 W/m2 for ERFari 2005-14 and -1.0 for ERFaci 2005-14.

# %%
# these come from AR6 WG1
# source: https://github.com/sarambl/AR6_CH6_RCMIPFIGS/blob/master/ar6_ch6_rcmipfigs/data_out/fig6_12_ts15_historic_delta_GSAT/2019_ERF_est.csv
# they sum to -0.22 W/m2, for 2019
# Calculate a radiative efficiency for each species from CEDS and updated concentrations.
df_ari_emitted_mean = pd.read_csv('../data/ar6/table_mean_thornhill_collins_orignames.csv', index_col=0)
erfari_emitted = pd.Series(df_ari_emitted_mean['Aerosol'])
erfari_emitted.rename_axis(None, inplace=True)
erfari_emitted.rename({'HC': 'EESC', 'VOC': 'NMVOC'}, inplace=True)
erfari_emitted

# %%
df_ari_emitted_std = pd.read_csv('../data/ar6/table_std_thornhill_collins_orignames.csv', index_col=0)
erfari_emitted_std = pd.Series(df_ari_emitted_std['Aerosol_sd'])
erfari_emitted_std.rename_axis(None, inplace=True)
erfari_emitted_std.rename({'HC': 'EESC', 'VOC': 'NMVOC'}, inplace=True)
erfari_emitted_std


# %%
def calculate_eesc(
    concentration,
    fractional_release,
    fractional_release_cfc11,
    cl_atoms,
    br_atoms,
    br_cl_ratio=45,
):

    # EESC is in terms of CFC11-eq
    eesc_out = (
        cl_atoms * (concentration) * fractional_release / fractional_release_cfc11
        + br_cl_ratio
        * br_atoms
        * (concentration)
        * fractional_release
        / fractional_release_cfc11
    ) * fractional_release_cfc11
    return eesc_out


fractional_release = {
    "CFC-11": 0.47,
    "CFC-12": 0.23,
    "CFC-113": 0.29,
    "CFC-114": 0.12,
    "CFC-115": 0.04,
    "HCFC-22": 0.13,
    "HCFC-141b": 0.34,
    "HCFC-142b": 0.17,
    "CCl4": 0.56,
    "CHCl3": 0,
    "CH2Cl2": 0,
    "CH3Cl": 0.44,
    "CH3CCl3": 0.67,
    "CH3Br": 0.6,
    "Halon-1211": 0.62,
    "Halon-1301": 0.28,
    "Halon-2402": 0.65,
}

cl_atoms = {
    "CFC-11": 3,
    "CFC-12": 2,
    "CFC-113": 3,
    "CFC-114": 2,
    "CFC-115": 1,
    "HCFC-22": 1,
    "HCFC-141b": 2,
    "HCFC-142b": 1,
    "CCl4": 4,
    "CHCl3": 3,
    "CH2Cl2": 2,
    "CH3Cl": 1,
    "CH3CCl3": 3,
    "CH3Br": 0,
    "Halon-1211": 1,
    "Halon-1301": 0,
    "Halon-2402": 0,
}

br_atoms = {
    "CFC-11": 0,
    "CFC-12": 0,
    "CFC-113": 0,
    "CFC-114": 0,
    "CFC-115": 0,
    "HCFC-22": 0,
    "HCFC-141b": 0,
    "HCFC-142b": 0,
    "CCl4": 0,
    "CHCl3": 0,
    "CH2Cl2": 0,
    "CH3Cl": 0,
    "CH3CCl3": 0,
    "CH3Br": 1,
    "Halon-1211": 1,
    "Halon-1301": 1,
    "Halon-2402": 2,
}

hc_eesc = {}
total_eesc = np.zeros(275)
for species in cl_atoms:
    hc_eesc[species] = calculate_eesc(
        concentrations.loc[:, species],
        fractional_release[species],
        fractional_release["CFC-11"],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc = total_eesc + hc_eesc[species]

total_eesc

# %%
#total_eesc = total_eesc.to_frame('EESC')

# %%
# erfari radiative efficiency per Mt or ppb or ppt
re = erfari_emitted / (emissions.loc[2019, :] - emissions.loc[1750, :])
re.dropna(inplace=True)

# %%
re['CH4'] = erfari_emitted['CH4'] / (concentrations.loc[2019, 'CH4'] - concentrations.loc[1750, 'CH4'])
re['N2O'] = erfari_emitted['N2O'] / (concentrations.loc[2019, 'N2O'] - concentrations.loc[1750, 'N2O'])
re['EESC'] = erfari_emitted['EESC'] / (total_eesc.loc[2019] - total_eesc.loc[1750])

# %%
re

# %%
re_std = erfari_emitted_std / (emissions.loc[2019, :] - emissions.loc[1750, :])
re_std.dropna(inplace=True)
re_std['CH4'] = erfari_emitted_std['CH4'] / (concentrations.loc[2019, 'CH4'] - concentrations.loc[1750, 'CH4'])
re_std['N2O'] = erfari_emitted_std['N2O'] / (concentrations.loc[2019, 'N2O'] - concentrations.loc[1750, 'N2O'])
re_std['EESC'] = erfari_emitted_std['EESC'] / (total_eesc.loc[2019] - total_eesc.loc[1750])
re_std

# %%
re.index

# %%
erfari_best = pd.concat(
    (
        (re * emissions)[['BC', 'OC', 'SO2', 'NOx', 'NMVOC', 'NH3']] - (re * emissions.loc[1750, ['BC', 'OC', 'SO2', 'NOx', 'NMVOC', 'NH3']]),
        (re * concentrations)[['CH4', 'N2O']] - (re * concentrations.loc[1750, ['CH4', 'N2O']]),
        re['EESC'] * (total_eesc - total_eesc.loc[1750])
    ), axis=1
).dropna(axis=1).sum(axis=1)

# %%
# 90% range of ERF uncertainty in 2019 from model estimates
np.sqrt((erfari_emitted_std**2).sum()) * NINETY_TO_ONESIGMA

# %%
# 90% range of ERF uncertainty in 2005-2014 from model estimates
(erfari_best.loc[2005:2014].mean()/-0.22) * np.sqrt((erfari_emitted_std**2).sum()) * NINETY_TO_ONESIGMA

# %%
# best estimate ERF in 2005-2014 from model estimates
erfari_best.loc[2005:2014].mean()

# %%
# we need to map the -0.27 +/- 0.57 to -0.3 +/- 0.3 which is the IPCC AR6 assessment
best_scale = -0.3 / erfari_best.loc[2005:2014].mean()
unc_scale = 0.3 / ((erfari_best.loc[2005:2014].mean()/-0.22) * np.sqrt((erfari_emitted_std**2).sum()) * NINETY_TO_ONESIGMA)

# %%
best_scale, unc_scale

# %%
forcing['aerosol-radiation_interactions'] = (erfari_best * best_scale).values

# %%
for specie in ['BC', 'OC', 'SO2', 'NOx', 'NMVOC', 'NH3']:
    print(specie, (emissions.loc[2024, specie] - emissions.loc[1750, specie]) * re[specie])


# %%
# convert to numpy for efficiency
erfari_re_samples = pd.DataFrame(
    scipy.stats.norm.rvs(
        re*best_scale, re_std*unc_scale, size=(SAMPLES, 9), random_state=3729329,
    ),
    columns = re.index
)[['BC', 'OC', 'SO2', 'NOx', 'NMVOC', 'NH3', 'CH4', 'N2O', 'EESC']]

# %%
erfari_re_samples

# %%
erfari_re_samples = erfari_re_samples.to_numpy()

# %%
# a future TODO is to split CO out from the NMVOC estimate
# then CH4 and N2O is produced only for the time series to bang into the emissions harmonization so drop them too
emnump = emissions.drop(columns=['CO', 'CH4', 'N2O']).to_numpy()

# %%
emissions

# %%
forcing_ensemble['aerosol-radiation_interactions'] = np.zeros((275, SAMPLES))
for i in tqdm(range(SAMPLES)):
    forcing_ensemble['aerosol-radiation_interactions'][:, i] = (
        (
            ((erfari_re_samples[i, :6] * emnump) - (erfari_re_samples[i, :6] * emnump[0, :])).sum(axis=1) + 
            ((erfari_re_samples[i, 6] * concentrations['CH4'].values) - (erfari_re_samples[i, 6] * concentrations.loc[1750, 'CH4'])) +
            ((erfari_re_samples[i, 7] * concentrations['N2O'].values) - (erfari_re_samples[i, 7] * concentrations.loc[1750, 'N2O'])) +
            (erfari_re_samples[i, 8] * (total_eesc.values - total_eesc.loc[1750]))
        )
    )

# %%
pl.plot(erfari_best * best_scale)

# %%
np.percentile(forcing_ensemble['aerosol-radiation_interactions'][255:265, :].mean(axis=0), (5, 50, 95))

# %%
df_aci_cal = pd.read_csv('../data/fair-calibrate-1.4.1/aerosol_cloud.csv', index_col=0)

# %%
df_aci_cal

# %%
beta_samp = df_aci_cal["aci_scale"]
n0_samp = df_aci_cal["Sulfur"]
n1_samp = df_aci_cal["BC"]
n2_samp = df_aci_cal["OC"]

# %%
np.log(n0_samp)

# %%
np.log(n1_samp)

# %%
np.log(n2_samp)

# %%
kde = scipy.stats.gaussian_kde([np.log(n0_samp), np.log(n1_samp), np.log(n2_samp)], bw_method=0.1)
aci_sample = kde.resample(size=SAMPLES * 1, seed=63648708)

# %%
#aci_sample[0, aci_sample[0, :] > 0] = 0#np.nan
#aci_sample[1, aci_sample[1, :] > 0] = 0#np.nan
#aci_sample[2, aci_sample[2, :] > 0] = 0#np.nan
# mask = np.any(np.isnan(aci_sample), axis=0)
# aci_sample = aci_sample[:, ~mask]

# %%
erfaci_sample = scipy.stats.norm.rvs(
    size=SAMPLES, loc=-1.0, scale=0.7/NINETY_TO_ONESIGMA, random_state=71271
)

# %%
so2 = emissions['SO2'].values
bc = emissions['BC'].values
oc = emissions['OC'].values

# %%
beta = np.zeros(SAMPLES)


# %%
def aci_log(x, beta, n0, n1, n2):
    aci = beta * np.log(1 + x[0] * n0 + x[1] * n1 + x[2] * n2)
    return aci


# %%
forcing_ensemble['aerosol-cloud_interactions'] = np.zeros((275, SAMPLES))
for i in tqdm(range(SAMPLES), desc="aci samples"):
    ts2010 = np.mean(
        aci_log(
            [so2[255:265], bc[255:265], oc[255:265]],
            1,
            np.exp(aci_sample[0, i]),
            np.exp(aci_sample[1, i]),
            np.exp(aci_sample[2, i]),
        )
    )
    ts1750 = aci_log(
        [so2[0], bc[0], oc[0]],
        1,
        np.exp(aci_sample[0, i]),
        np.exp(aci_sample[1, i]),
        np.exp(aci_sample[2, i]),
    )
    forcing_ensemble['aerosol-cloud_interactions'][:, i] = (
       (
           aci_log(
               [so2, bc, oc],
               1,
               np.exp(aci_sample[0, i]),
               np.exp(aci_sample[1, i]),
               np.exp(aci_sample[2, i]),
           )
           - ts1750
       )
       / (ts2010 - ts1750)
       * (erfaci_sample[i])
    )
    beta[i] = erfaci_sample[i] / (ts2010 - ts1750)

# %%
#pl.plot(erfaci);
#pl.ylim(-6, 0.2)

# %%
(forcing_ensemble['aerosol-cloud_interactions'][255:265, :]).mean()

# %%
np.median(forcing_ensemble['aerosol-cloud_interactions'][255:265, :].mean(axis=0))

# %%
forcing['aerosol-cloud_interactions'] = np.median(forcing_ensemble['aerosol-cloud_interactions'], axis=1)
pl.plot(forcing['aerosol-cloud_interactions'])

# %%
pl.plot(forcing['aerosol-radiation_interactions'] + forcing['aerosol-cloud_interactions'])

# %% [markdown]
# ### Contrail forcing
#
# Use result from previous notebook

# %%
df_contrails = pd.read_csv('../output/contrails_ERF_1930-2024.csv', index_col=0)
forcing['contrails'] = np.zeros(275)
forcing['contrails'][180:] = df_contrails.values.squeeze()

# %% [markdown]
# ### Land use forcing
#
# Now: take separate time series for land use and irrigation
#
# Also compare to cumulative land use CO2 emissions, scale to -0.2 W/m2 for 1750 to 2019 (the old way of doing things)

# %%
df_gcp = pd.read_csv('../data/gcp_emissions/gcp_2024.csv', index_col=0)
df_luprocess = pd.read_csv('../output/land_use_1750-2024.csv', index_col=0)

# %%
df_gcp['AFOLU']

# %%
df_luprocess

# %%
lusf2019 = -0.20/(np.cumsum(df_gcp['AFOLU']).loc[2019] - df_gcp.loc[1750, 'AFOLU'])
lusf2019

# %%
forcing['land_use'] = df_luprocess['total'].values
landuse_from_co2_afolu = (np.cumsum(df_gcp['AFOLU']) - df_gcp.loc[1750, 'AFOLU']).values*lusf2019

# %%
pl.plot(landuse_from_co2_afolu)
pl.plot(forcing['land_use'])

# %% [markdown]
# ## BC on snow
#
# linear with emissions, 2019 ERF = 0.08

# %%
emissions.loc[2019,'BC']
forcing['BC_on_snow'] = (0.08*(emissions['BC']-emissions.loc[1750,'BC'])/(emissions.loc[2019,'BC']-emissions.loc[1750,'BC'])).values.squeeze()
pl.plot(forcing['BC_on_snow'])

# %% [markdown]
# ### Greenhouse gas concentrations
#
# Here, tropospheric and surface adjustments are only implemented for CO2, CH4, N2O, CFC11 and CFC12 to convert SARF to ERF. There's an argument to uplift ERF by 5% for other GHGs based on land surface warming, but the total forcing will be very small and no single-forcing studies exist. This was not done in AR6 chapter 7.
#
# Radiative efficiencies for F-gases are from Hodnebrog et al. 2020 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019RG000691.

# %%
concentrations

# %%
meinshausen2020

# %%
concentrations.loc[2024].values.shape

# %%
# radiative efficiencies
# source: Hodnebrog et al 2020 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019RG000691
radeff = {
    'HFC-125':      0.23378,
    'HFC-134a':     0.16714,
    'HFC-143a':     0.168,
    'HFC-152a':     0.10174,
    'HFC-227ea':    0.27325,
    'HFC-23':       0.19111,
    'HFC-236fa':    0.25069,
    'HFC-245fa':    0.24498,
    'HFC-32':       0.11144,
    'HFC-365mfc':   0.22813,
    'HFC-43-10mee': 0.35731,
    'NF3':          0.20448,
    'C2F6':         0.26105,
    'C3F8':         0.26999,
    'n-C4F10':      0.36874,
    'n-C5F12':      0.4076,
    'n-C6F14':      0.44888,
    'i-C6F14':      0.44888,
    'C7F16':        0.50312,
    'C8F18':        0.55787,
    'CF4':          0.09859,
    'c-C4F8':       0.31392,
    'SF6':          0.56657,
    'SO2F2':        0.21074,
    'CCl4':         0.16616,
    'CFC-11':       0.25941,
    'CFC-112':      0.28192,
    'CFC-112a':     0.24564,
    'CFC-113':      0.30142,
    'CFC-113a':     0.24094, 
    'CFC-114':      0.31433,
    'CFC-114a':     0.29747,
    'CFC-115':      0.24625,
    'CFC-12':       0.31998,
    'CFC-13':       0.27752,
    'CH2Cl2':       0.02882,
    'CH3Br':        0.00432,
    'CH3CCl3':      0.06454,
    'CH3Cl':        0.00466,
    'CHCl3':        0.07357,
    'HCFC-124':     0.20721,
    'HCFC-133a':    0.14995,
    'HCFC-141b':    0.16065,
    'HCFC-142b':    0.19329,
    'HCFC-22':      0.21385,
    'HCFC-31':      0.068,
    'Halon-1202':   0,       # not in dataset
    'Halon-1211':   0.30014,
    'Halon-1301':   0.29943,
    'Halon-2402':   0.31169,
    'CO2':          0,       # different relationship
    'CH4':          0,       # different relationship
    'N2O':          0        # different relationship
}

radeff_array = np.ones(52) * np.nan
for igas, gas in enumerate(concentrations.columns):
    radeff_array[igas] = radeff[gas]

# %%
np.where(concentrations.columns=='CFC-11')[0][0], np.where(concentrations.columns=='CFC-12')[0][0]

# %%
adjustments = np.ones(52)
adjustments[0] = 1.05
adjustments[1] = 0.86
adjustments[2] = 1.07
adjustments[21] = 1.12
adjustments[22] = 1.13

# %%
meinshausen2020(
    concentrations.loc[2019].values,
    concentrations.loc[1750].values,
    adjustments,
    radeff_array,
    [0],
    [1],
    [2],
    list(range(3,52))
)

# %%
ghg_out = np.zeros((275, 52))
for i, year in enumerate(range(1750, 2025)):
    ghg_out[i, :] = meinshausen2020(
        concentrations.loc[year].values,
        concentrations.loc[1750].values,
        adjustments,
        radeff_array,
        [0],
        [1],
        [2],
        list(range(3,52))
    )
for igas, gas in enumerate(concentrations.columns):
    forcing[gas] = ghg_out[:, igas]

# %%
pl.plot(ghg_out.sum(axis=1))
pl.plot(forcing['CO2'])

# %%
forcing['CO2'][0]

# %% [markdown]
# ### Ozone
#
# Same as AR6 method: use ERF time series from Skeie et al. (2020) from 6 CMIP6 models, and fit coefficients for each precursor based on Thornhill et al. (2021)

# %%
# get temperature time series, needed to back out temperature feedback
temp_obs = pd.read_csv('../data/gmst/IGCC_GMST_1850-2024.csv', index_col=0).values
delta_gmst = [
    0,
    temp_obs[65:76].mean(),
    temp_obs[75:86].mean(),
    temp_obs[85:96].mean(),
    temp_obs[95:106].mean(),
    temp_obs[105:116].mean(),
    temp_obs[115:126].mean(),
    temp_obs[125:136].mean(),
    temp_obs[135:146].mean(),
    temp_obs[145:156].mean(),
    temp_obs[152:163].mean(),
    temp_obs[155:166].mean(),
    temp_obs[159:170].mean(),
    temp_obs[167].mean(),  # we don't use this
    temp_obs[168].mean(),
]  # this is the time periods used in Skeie et al.

# %%
# get the skeie data and back out the climate feedback (-0.037 W/m2/K)
good_models = [
    "BCC-ESM1",
    "CESM2(WACCM6)",
    "GFDL-ESM4",
    "GISS-E2-1-H",
    "MRI-ESM2-0",
    "OsloCTM3",
]

skeie_trop = pd.read_csv(
    "../data/ozone/skeie_ozone_trop.csv", index_col=0
)
skeie_trop = skeie_trop.loc[good_models]
skeie_trop.insert(0, 1850, 0)
skeie_trop.columns = pd.to_numeric(skeie_trop.columns)
skeie_trop.interpolate(axis=1, method="values", limit_area="inside", inplace=True)

skeie_strat = pd.read_csv(
    "../data/ozone/skeie_ozone_strat.csv", index_col=0
)
skeie_strat = skeie_strat.loc[good_models]
skeie_strat.insert(0, 1850, 0)
skeie_strat.columns = pd.to_numeric(skeie_strat.columns)
skeie_strat.interpolate(axis=1, method="values", limit_area="inside", inplace=True)

skeie_total = skeie_trop + skeie_strat

coupled_models = copy.deepcopy(good_models)
coupled_models.remove("OsloCTM3")

skeie_total.loc[coupled_models] = skeie_total.loc[coupled_models] - (-0.037) * np.array(
    delta_gmst
)
skeie_ssp245 = skeie_total.mean()
skeie_ssp245[1750] = -0.03
skeie_ssp245.sort_index(inplace=True)
skeie_ssp245 = skeie_ssp245 + 0.03
skeie_ssp245.drop([2014, 2017, 2020], inplace=True)
# skeie_ssp245 = skeie_ssp245.append(
#     skeie_total.loc["OsloCTM3", 2014:]
#     - skeie_total.loc["OsloCTM3", 2010]
#     + skeie_ssp245[2010]
# )
skeie_ssp245 = pd.concat(
    (
        skeie_ssp245,
        skeie_total.loc["OsloCTM3", 2014:] - skeie_total.loc["OsloCTM3", 2010] + skeie_ssp245[2010]
    )
)

f = interp1d(
    skeie_ssp245.index, skeie_ssp245, bounds_error=False, fill_value="extrapolate"
)
years = np.arange(1750, 2021)
o3total = f(years)

# %%
# Get 1750-2014 changes to compare with Thornhill CMIP6
# Thornhill uses 1850-2014. However, we want a 1750 baseline, and the coefficients will be scaled up anyway in order
# to match the Thornhill best estimate (0.31) to Skeie (0.47)
delta_Cch4 = concentrations.loc[2014, 'CH4'] - concentrations.loc[1750, 'CH4']
delta_Cn2o = concentrations.loc[2014, 'N2O'] - concentrations.loc[1750, 'N2O']
delta_Cods = total_eesc.loc[2014] - total_eesc.loc[1750]
delta_Eco = emissions.loc[2014, 'CO'] - emissions.loc[1750, 'CO']
delta_Enox = emissions.loc[2014, 'NOx'] - emissions.loc[1750, 'NOx']
delta_Evoc = emissions.loc[2014, 'NMVOC'] - emissions.loc[1750, 'NMVOC']

# %%
# create a timeseries of precursors as a new array
ts = np.vstack(
    (
        concentrations["CH4"],
        concentrations["N2O"],
        total_eesc,
        emissions["CO"],
        emissions["NMVOC"],
        emissions["NOx"],
    )
).T

# %%
# best estimate radiative efficienices from 2014 - 1850 from AR6 here from Thornhill. Used for scaling.
radeff_ch4 = 0.14 / delta_Cch4
radeff_n2o = 0.03 / delta_Cn2o
radeff_ods = -0.11 / delta_Cods  # excludes UKESM
radeff_co = 0.067 / delta_Eco  # stevenson CMIP5 scaled to CO + VOC total
radeff_voc = 0.043 / delta_Evoc  # stevenson CMIP5 scaled to CO + VOC total
radeff_nox = 0.20 / delta_Enox

# %%
# scale factor to convert from Skeie to Thornhill total ozone ERF
fac_cmip6_skeie = (
    radeff_ch4 * delta_Cch4
    + radeff_n2o * delta_Cn2o
    + radeff_ods * delta_Cods
    + radeff_co * delta_Eco
    + radeff_voc * delta_Evoc
    + radeff_nox * delta_Enox
) / (o3total[264] - o3total[0])
fac_cmip6_skeie


# %%
# Fit timeseries of ozone forcing in the absence of climate feedbacks (calculated above)
# This is to compare the fixed-SST runs from Thornhill with the coupled runs from Skeie
# We back out the warming contribution from Skeie
# The bounds of the curve fit are a 90% range from model results in Thornhill et al. 2021, except for halocarbons where
# we exclude UKESM
# Thornhill coefficients are scaled (fac_cmip6_skeie) to preserve total ERF
def fit_precursors(x, rch4, rn2o, rods, rco, rvoc, rnox):
    return (
        rch4 * x[0] + rn2o * x[1] + rods * x[2] + rco * x[3] + rvoc * x[4] + rnox * x[5]
    )

p, cov = curve_fit(
    fit_precursors,
    ts[:270, :].T - ts[0:1, :].T,
    o3total[:270] - o3total[0],
    bounds=(  # 90% range from Thornhill for each precursor
        (
            0.09 / delta_Cch4 / fac_cmip6_skeie,
            0.01 / delta_Cn2o / fac_cmip6_skeie,
            -0.21 / delta_Cods / fac_cmip6_skeie,
            0.010 / delta_Eco / fac_cmip6_skeie,
            0 / delta_Evoc / fac_cmip6_skeie,
            0.09 / delta_Enox / fac_cmip6_skeie,
        ),
        (
            0.19 / delta_Cch4 / fac_cmip6_skeie,
            0.05 / delta_Cn2o / fac_cmip6_skeie,
            -0.01 / delta_Cods / fac_cmip6_skeie,
            0.124 / delta_Eco / fac_cmip6_skeie,
            0.086 / delta_Evoc / fac_cmip6_skeie,
            0.31 / delta_Enox / fac_cmip6_skeie,
        ),
    ),
)


# %%
forcing['O3'] = (
    p[0] * (concentrations["CH4"] - concentrations.loc[1750, "CH4"])
    + p[1] * (concentrations["N2O"] - concentrations.loc[1750, "N2O"])
    + p[2] * (total_eesc - total_eesc.loc[1750])
    + p[3] * (emissions["CO"] - emissions.loc[1750, "CO"])
    + p[4] * (emissions["NMVOC"] - emissions.loc[1750, "NMVOC"])
    + p[5] * (emissions["NOx"] - emissions.loc[1750, "NOx"])
).values.squeeze()

# %%
pl.plot(np.arange(1750, 2025), forcing['O3'], label='ozone precursor best fit')
pl.plot(np.arange(1750, 2021), o3total, label='Skeie et al. (historical + SSP2-4.5)')
pl.legend()

# %% [markdown]
# ### Stratospheric water vapour
#
# Simple scaling with methane concentration delta. Anchored to be 0.05 W/m2 in 2019.

# %%
sfh2ostrat = 0.05 / (concentrations['CH4'][2019] - concentrations['CH4'][1750])
forcing['H2O_stratospheric'] = ((concentrations['CH4'] - concentrations['CH4'][1750]) * sfh2ostrat)

# %%
forcing['H2O_stratospheric']

# %% [markdown]
# ## Aggregation

# %%
type(forcing['H2O_stratospheric']) == pd.core.series.Series

# %%
pd.DataFrame(forcing, index=np.arange(1750, 2025)).to_csv('../output/ERF_best_1750-2024.csv')

# %%
# aggregate land use first
irr_ensemble = df_luprocess['irrigation'].values[:, None] * scale_df['irrigation'].values[None,:]
lu_ensemble = df_luprocess['LUH2-GCB2024_rescaled'].values[:, None] * scale_df['land_albedo'].values[None,:]

# %%
forcing_ensemble['land_use'] = irr_ensemble + lu_ensemble

# %%
for agent in tqdm(forcing):
    if agent not in ['aerosol-radiation_interactions', 'aerosol-cloud_interactions', 'solar', 'land_use']:
        if type(forcing[agent]) == pd.core.series.Series:
            forcing[agent] = forcing[agent].values
        forcing_ensemble[agent] = forcing[agent][:,None] * scale_df[agent].values[None,:]

# %%
forcing_ensemble

# %%
forcing_ensemble_sum = np.zeros((275, SAMPLES))
forcing_ensemble_anthro = np.zeros((275, SAMPLES))
forcing_ensemble_natural = np.zeros((275, SAMPLES))
forcing_ensemble_aerosol = np.zeros((275, SAMPLES))
forcing_ensemble_minorghg = np.zeros((275, SAMPLES))

for agent in tqdm(forcing):
    forcing_ensemble_sum = forcing_ensemble_sum + forcing_ensemble[agent]
    if agent in ['solar', 'volcanic']:
        forcing_ensemble_natural = forcing_ensemble_natural + forcing_ensemble[agent]
    else:
        forcing_ensemble_anthro = forcing_ensemble_anthro + forcing_ensemble[agent]
    if agent not in ['solar', 'volcanic', 'aerosol-radiation_interactions', 'aerosol-cloud_interactions',
                    'CO2', 'CH4', 'N2O', 'O3', 'H2O_stratospheric', 'contrails', 'BC_on_snow', 'land_use']:
        forcing_ensemble_minorghg = forcing_ensemble_minorghg + forcing_ensemble[agent]
    if agent in ['aerosol-radiation_interactions', 'aerosol-cloud_interactions']:
        forcing_ensemble_aerosol = forcing_ensemble_aerosol + forcing_ensemble[agent]

# %%
forcing_sum = np.zeros((275))
forcing_anthro = np.zeros((275))
forcing_natural = np.zeros((275))
forcing_aerosol = np.zeros((275))
forcing_minorghg = np.zeros((275))

for agent in tqdm(forcing):
    forcing_sum = forcing_sum + forcing[agent]
    if agent in ['solar', 'volcanic']:
        forcing_natural = forcing_natural + forcing[agent]
    else:
        forcing_anthro = forcing_anthro + forcing[agent]
    if agent not in ['solar', 'volcanic', 'aerosol-radiation_interactions', 'aerosol-cloud_interactions',
                    'CO2', 'CH4', 'N2O', 'O3', 'H2O_stratospheric', 'contrails', 'BC_on_snow', 'land_use']:
        forcing_minorghg = forcing_minorghg + forcing[agent]
    if agent in ['aerosol-radiation_interactions', 'aerosol-cloud_interactions']:
        forcing_aerosol = forcing_aerosol + forcing[agent]

# %%
forcing_p05 = {}
forcing_p95 = {}

for agent in tqdm(forcing):
    forcing_p05[agent] = np.percentile(forcing_ensemble[agent],5,axis=1)
    forcing_p95[agent] = np.percentile(forcing_ensemble[agent],95,axis=1)
    
forcing_p05_sum = np.percentile(forcing_ensemble_sum,5,axis=1)
forcing_p05_anthro = np.percentile(forcing_ensemble_anthro,5,axis=1)
forcing_p05_natural = np.percentile(forcing_ensemble_natural,5,axis=1)
forcing_p05_aerosol = np.percentile(forcing_ensemble_aerosol,5,axis=1)
forcing_p05_minorghg = np.percentile(forcing_ensemble_minorghg,5,axis=1)

forcing_p95_sum = np.percentile(forcing_ensemble_sum,95,axis=1)
forcing_p95_anthro = np.percentile(forcing_ensemble_anthro,95,axis=1)
forcing_p95_natural = np.percentile(forcing_ensemble_natural,95,axis=1)
forcing_p95_aerosol = np.percentile(forcing_ensemble_aerosol,95,axis=1)
forcing_p95_minorghg = np.percentile(forcing_ensemble_minorghg,95,axis=1)

# %%
fig, ax = pl.subplots(4,4, figsize=(16,16),squeeze=True)
ax[0,0].fill_between(np.arange(1750,2025), forcing_p05['CO2'], forcing_p95['CO2'], alpha=0.3)
ax[0,0].plot(np.arange(1750,2025),forcing['CO2'])
ax[0,0].set_title('CO2')
ax[0,1].fill_between(np.arange(1750,2025), forcing_p05['CH4'], forcing_p95['CH4'], alpha=0.3)
ax[0,1].plot(np.arange(1750,2025),forcing['CH4'])
ax[0,1].set_title('CH4')
ax[0,2].fill_between(np.arange(1750,2025), forcing_p05['N2O'], forcing_p95['N2O'], alpha=0.3)
ax[0,2].plot(np.arange(1750,2025),forcing['N2O'])
ax[0,2].set_title('N2O')
ax[0,3].fill_between(np.arange(1750,2025), forcing_p05_minorghg, forcing_p95_minorghg, alpha=0.3)
ax[0,3].plot(np.arange(1750,2025),forcing_minorghg)
ax[0,3].set_title('Other WMGHGs')
ax[1,0].fill_between(np.arange(1750,2025), forcing_p05['O3'], forcing_p95['O3'], alpha=0.3)
ax[1,0].plot(np.arange(1750,2025),forcing['O3'])
ax[1,0].set_title('O3')
ax[1,1].fill_between(np.arange(1750,2025), forcing_p05['H2O_stratospheric'], forcing_p95['H2O_stratospheric'], alpha=0.3)
ax[1,1].plot(np.arange(1750,2025),forcing['H2O_stratospheric'])
ax[1,1].set_title('H2O stratospheric')
ax[1,2].fill_between(np.arange(1750,2025), forcing_p05['contrails'], forcing_p95['contrails'], alpha=0.3)
ax[1,2].plot(np.arange(1750,2025),forcing['contrails'])
ax[1,2].set_title('contrails')
ax[1,3].fill_between(np.arange(1750,2025), forcing_p05['aerosol-radiation_interactions'], forcing_p95['aerosol-radiation_interactions'], alpha=0.3)
ax[1,3].plot(np.arange(1750,2025),forcing['aerosol-radiation_interactions'])
ax[1,3].set_title('ERFari')
ax[2,0].fill_between(np.arange(1750,2025), forcing_p05['aerosol-cloud_interactions'], forcing_p95['aerosol-cloud_interactions'], alpha=0.3)
ax[2,0].plot(np.arange(1750,2025),forcing['aerosol-cloud_interactions'])
ax[2,0].set_title('ERFaci')
ax[2,1].fill_between(np.arange(1750,2025), forcing_p05['BC_on_snow'], forcing_p95['BC_on_snow'], alpha=0.3)
ax[2,1].plot(np.arange(1750,2025),forcing['BC_on_snow'])
ax[2,1].set_title('BC on snow')
ax[2,2].fill_between(np.arange(1750,2025), forcing_p05['land_use'], forcing_p95['land_use'], alpha=0.3)
ax[2,2].plot(np.arange(1750,2025),forcing['land_use'])
ax[2,2].set_title('land use')
ax[2,3].fill_between(np.arange(1750,2025), forcing_p05['volcanic'], forcing_p95['volcanic'], alpha=0.3)
ax[2,3].plot(np.arange(1750,2025),forcing['volcanic'])
ax[2,3].set_title('volcanic')
ax[3,0].fill_between(np.arange(1750,2025), forcing_p05['solar'], forcing_p95['solar'], alpha=0.3)
ax[3,0].plot(np.arange(1750,2025),forcing['solar'])
ax[3,0].set_title('solar');

# %%
pl.figure(figsize=(16,9))
pl.fill_between(np.arange(1750,2025), forcing_p05_sum, forcing_p95_sum, alpha=0.3)
pl.plot(np.arange(1750,2025), forcing_sum)
pl.yticks(np.arange(-4,5))
pl.xticks(np.arange(1750,2020,50))
pl.ylim(-4,4)
pl.xlim(1750,2025)
pl.grid()

# %%
forcing_ensemble

# %%
forcing_ensemble['CO2']

# %%
for agent in forcing_ensemble:
    forcing_ensemble[agent] = xr.DataArray(
        forcing_ensemble[agent], 
        coords=dict(
            time=np.arange(1750, 2025),
            ensemble=np.arange(SAMPLES)
        )
    )

# %%
xr.Dataset(forcing_ensemble).to_netcdf('../output/ERF_ensemble.nc')

# %%
