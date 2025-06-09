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
# # Calculate GHG concentrations
#
# **update Jan 2025: while we wait for Brad and Lindsay's assessments, we will extrapolate last year's numbers for the preliminary WMO data.**
#
# **update Feb 2025: preliminary 2024 values from Lindsay in the paper draft, use these here for WMO analysis**
#
# **update Apr 2025: 2024 NOAA CO2 value now available, additional dataset for some minor GHGs from NOAA and AGAGE**
#
# IPCC AR6 methodology:
#
# The following description comes from the Excel sheet of long-lived greenhouse gas concentrations, v9. See https://github.com/chrisroadmap/ar6/blob/main/data_input/observations/LLGHG_history_AR6_v9_for_archive.xlsx
#
# All values are mid-year mean.
#
# Xin Lan and Brad Hall updated most GHGs to 2023. All others are extrapolated from most recent year - either NOAA, AGAGE, or value used in IPCC.
#
# https://gml.noaa.gov/aftp/data/ is usually a good place to look
#
# NOAA (accessed 2025-06-09):
# - https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt
# - https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_annmean_gl.txt
# - https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_annmean_gl.txt
# - https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_annmean_gl.txt
# - https://gml.noaa.gov/aftp/data/hats/Total_Cl_Br/2024%20update%20total%20Cl%20Br%20&%20F.xls  (converted to CSV with header and footer rows stripped out; save as noaa_**YYYY**_global_mean_mixing_ratios.csv) **note: each year, check the FTP directory to see if there has been an annual update**
#
# AGAGE (accessed 2025-02-12, no update as of June):
# - https://agage2.eas.gatech.edu/data_archive/global_mean/global_mean_ms.txt
# - https://agage2.eas.gatech.edu/data_archive/global_mean/global_mean_md.txt
#
# Then:
# - CSIRO for CO2, CH4 and N2O. These values come from Paul Krummel. (We comment on CSIRO in the paper but don't use them in the assessment)
# - AGAGE "horse's mouth" figures from Jens Muhle; CCl4, CFC-11, CFC-12, CH4, HCFC-22, HFC-125 and N2O

# %%
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit

# %%
pd.set_option('display.max_columns', 500)

# %%
df_co2 = pd.read_csv(
    '../data/ghg_concentrations/noaa_gml/co2_annmean_gl.txt', 
    sep=r'\s+',
    comment='#', 
    names=['year', 'mean', 'unc'],
    index_col=0
)

# %%
df_ch4_noaa = pd.read_csv(
    '../data/ghg_concentrations/noaa_gml/ch4_annmean_gl.txt', 
    sep=r'\s+',
    comment='#', 
    names=['year', 'mean', 'unc'],
    index_col=0
)

# %%
df_n2o_noaa = pd.read_csv(
    '../data/ghg_concentrations/noaa_gml/n2o_annmean_gl.txt', 
    sep=r'\s+',
    comment='#', 
    names=['year', 'mean', 'unc'],
    index_col=0
)

# %%
df_noaa = pd.read_csv(
    '../data/ghg_concentrations/noaa_gml/noaa_2024_global_mean_mixing_ratios.csv'
)

df_noaa
df_noaa.drop(columns=["Unnamed: 20", "in ppt", "Unnamed: 22", "in ppt.1", "Unnamed: 24", "in ppt.2"], inplace=True)
df_noaa

df_noaa[df_noaa=="ND"]=np.nan

df_noaa = df_noaa.rolling(6, center=True).mean()
df_noaa['YYYY'] = (df_noaa.date-0.5)
df_noaa.drop(df_noaa.tail(2).index,inplace=True)
df_noaa.drop(df_noaa.head(3).index,inplace=True)
df_noaa.set_index('YYYY', inplace=True)
df_noaa.drop(columns=['date'], inplace=True)
df_noaa.rename(columns={'H2402': 'H-2402'}, inplace=True)
df_noaa = df_noaa[df_noaa.index % 1 == 0]
df_noaa.index = df_noaa.index.astype(int)
df_noaa

# %%
df_agage_ms = pd.read_csv(
    '../data/ghg_concentrations/agage/global_mean_ms.txt', 
    sep=r'\s+',
    skiprows=14,
    index_col=0
)

# %%
df_agage_ms

# %%
df_agage_ms = df_agage_ms.rolling(12, center=True).mean().drop([col for col in df_agage_ms.columns if '---' in col],axis=1)
df_agage_ms.drop(columns='MM', inplace=True)
df_agage_ms.set_index('YYYY', inplace=True)
df_agage_ms = df_agage_ms[df_agage_ms.index % 1 == 0]
df_agage_ms.index = df_agage_ms.index.astype(int)

# %%
df_agage_ms[df_agage_ms.index % 1 == 0]

# %%
df_conc = pd.read_csv(
    '../data/ghg_concentrations/ar6_updated/ipcc_ar6_wg1.csv',
    index_col=0
)

# %%
# insert rows from 2019 up to and including assessment year
df_conc.loc[2020, :] = np.nan
df_conc.loc[2021, :] = np.nan
df_conc.loc[2022, :] = np.nan
df_conc.loc[2023, :] = np.nan
df_conc.loc[2024, :] = np.nan

# %%
df_conc.tail(11)

# %%
df_co2

# %%
# this comes from Brad and Lindsay and was used directly in IGCC2023
df_update = pd.read_csv(
    '../data/ghg_concentrations/ar6_updated/update_2019-2023.csv',
    index_col=0
)

# note CO2 is on X2007 scale in the 2022 Climate Indicators - the offset is 0.18 ppm
df_update

# %%
# typically year of assessment minus 1
df_ch4_noaa.loc[1984:2024, 'mean']

# %%
# this comes from Jens Muhle for IGCC2024
df_agage_recent = pd.read_csv(
    '../data/ghg_concentrations/agage/agage_2019-2024.csv',
    index_col=0
)

# %%
df_agage_recent

# %%
# this comes from Lindsay Lan for IGCC2024

# for CO2, preference is given to the GML data online as it is more recently updated and given to two decimal places.
# for other gases, use this dataset.
df_noaa_update = pd.read_excel(
    '../data/ghg_concentrations/noaa_gml/NOAA_Annual_Mean_MoleFractions_with2024.xlsx',
    skiprows=2,
    skipfooter=5,
    index_col=0
)
df_noaa_update.drop(index=np.nan, inplace=True)
df_noaa_update.index = (df_noaa_update.index-0.5).astype(int)  # the index provided by Lindsay is correct but we want to be consistent across datasets
df_noaa_update[df_noaa_update=="nd"]=np.nan
df_noaa_update.rename(columns={'H2402': 'H-2402'}, inplace=True)

# NB: some of the GHG concentrations are marked red in the excel spreadsheet, and they do look a bit fishy. In this update I will exclude them 
# and see what impact it makes

# CFC-11 is marked red but the trend looks OK - it follows recent declines
# CCl4: the same
# CFC-12: the same
# Halon-1301 has an increase after a trend of stabilisation and decline. Maybe stick with this
df_noaa_update.loc[2024, "H-1301"] = np.nan
# HFC-143a shows a decline. This probably isn't right
df_noaa_update.loc[2024, "HFC-143a"] = np.nan
# HFC-125 the same
df_noaa_update.loc[2024, "HFC-125"] = np.nan

df_noaa_update

# %%
# decision tree
# CO2 is NOAA only, from GML (higher precision and newer than Lindsay's file)
# CO2: this is on X2019 scale. Meinshausen et al (2017) is the source of pre-1980 data in AR6. 
# Meinshausen data should be adjusted for the X2019 scale.
# X2019 = 1.00079*X2007 - 0.142 (from Brad)
df_conc.loc[1750:1978, 'CO2'] = df_conc.loc[1750:1978, 'CO2'] * 1.00079 - 0.142
df_conc.loc[1979:2024, 'CO2'] = df_co2.loc[1979:2024, 'mean']

# CH4 and N2O is average of NOAA (Lindsay's file) and AGAGE
# same for CCl4, CFC-11, CFC-12, CH4, HCFC-22, HFC-134a
# For methane and N2O, the calibration scales have not changed, and we use multiple datasets.
# since we don't have the AGAGE data here before 2019, take the average of NOAA and AGAGE from 2019 onwards and keep pre-2019 from IPCC
# HFC-125 is now done differently
for gas in ['CCl4', 'CFC-11', 'CFC-12', 'CH4', 'HCFC-22', 'HFC-134a', 'N2O']:
    df_conc.loc[2019:2024, gas] = np.mean((df_noaa_update.loc[2019:2024, gas], df_agage_recent.loc[2019:2024, gas]), axis=0).astype(float)

# %%
[col for col in df_noaa_update.columns if col in df_agage_ms.columns]

# %%
df_noaa_update.rename(columns={'H-2402': 'Halon-2402'}, inplace=True)
df_noaa_update.rename(columns={'H-1211': 'Halon-1211'}, inplace=True)
df_noaa_update.rename(columns={'H-1301': 'Halon-1301'}, inplace=True)

# %%
df_noaa.rename(columns={'H-2402': 'Halon-2402'}, inplace=True)
df_noaa.rename(columns={'H-1211': 'Halon-1211'}, inplace=True)
df_noaa.rename(columns={'H-1301': 'Halon-1301'}, inplace=True)

# %%
df_agage_ms.rename(columns={'H-2402': 'Halon-2402'}, inplace=True)
df_agage_ms.rename(columns={'H-1211': 'Halon-1211'}, inplace=True)
df_agage_ms.rename(columns={'H-1301': 'Halon-1301'}, inplace=True)

# %%
fig, ax = pl.subplots(4,4, figsize=(16, 16))
for igas, gas in enumerate(['HCFC-22', 'CFC-113', 'HCFC-141b', 'HCFC-142b', 'CH3CCl3', 'Halon-1211', 'Halon-1301', 'Halon-2402',
    'HFC-152a','HFC-143a', 'HFC-365mfc', 'HFC-227ea', 'HFC-23', 'SF6']):
    i = igas//4
    j = igas%4
    ax[i,j].plot(df_conc[gas])
    ax[i,j].plot(df_noaa_update[gas])
    ax[i,j].plot(df_agage_ms[gas])
    ax[i,j].set_title(gas)
    ax[i,j].set_xlim(1980, 2025)

# %%
# remove unreliable data
df_agage_ms.loc[2004:2007,'HFC-23'] = np.nan
df_agage_ms.loc[2018:,'CFC-113'] = np.nan
df_agage_ms.loc[2004:2005,'HFC-365mfc'] = np.nan
df_agage_ms.loc[2004:2006,'HFC-227ea'] = np.nan
df_noaa_update.loc[:2006, 'HFC-227ea'] = np.nan

# %%
# replot with questionable data removed
fig, ax = pl.subplots(4,4, figsize=(16, 16))
for igas, gas in enumerate(['HCFC-22', 'CFC-113', 'HCFC-141b', 'HCFC-142b', 'CH3CCl3', 'Halon-1211', 'Halon-1301', 'Halon-2402',
    'HFC-152a','HFC-143a', 'HFC-365mfc', 'HFC-227ea', 'HFC-23', 'SF6']):
    i = igas//4
    j = igas%4
    ax[i,j].plot(df_conc[gas])
    ax[i,j].plot(df_noaa_update[gas])
    ax[i,j].plot(df_agage_ms[gas])
    ax[i,j].set_title(gas)
    ax[i,j].set_xlim(1980, 2025)

# %%
# make a combined timeseries which is the mean of NOAA and AGAGE in 2020, 2021 and 2022, and
# NOAA 2023 and 2024 is offset with the mean of NOAA and AGAGE in all overlapping years

# turn off pandas warnings - why is pandas so fragile? why do they keep changing stuff?!
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for gas in ['HCFC-22', 'HCFC-141b', 'HCFC-142b', 'CH3CCl3', 'Halon-1211', 'Halon-1301', 'Halon-2402',
        'HFC-152a','HFC-143a', 'HFC-365mfc', 'HFC-227ea', 'HFC-23', 'SF6']:
        two_dataset_mean = pd.DataFrame((df_noaa_update[gas] - df_agage_ms[gas])).mean().values[0]
        df_conc.loc[2020:2022, gas] = pd.DataFrame((df_noaa_update.loc[2020:2022, gas], df_agage_ms.loc[2020:2022, gas])).mean()
        df_conc.loc[2023:2024, gas] = pd.DataFrame((df_noaa_update.loc[2023:2024, gas]) - two_dataset_mean)

# %%
# since we removed HFC-125, add it back here

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for gas in ['HFC-125']:
        two_dataset_mean = pd.DataFrame((df_noaa_update[gas] - df_agage_recent[gas])).mean().values[0]
        df_conc.loc[2020:2023, gas] = pd.DataFrame((df_noaa_update.loc[2020:2023, gas], df_agage_recent.loc[2020:2023, gas])).mean()
        df_conc.loc[2024, gas] = df_agage_recent.loc[2024, gas] - two_dataset_mean

# %%
# CFC-113 is special case; do the NOAA extrapolation from 2020, and assume the previously given values for 2018 and 2019 are good.
# we will check this in a new round of plots.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    gas = 'CFC-113'
    two_dataset_mean = pd.DataFrame((df_noaa_update[gas] - df_agage_ms[gas])).mean().values[0]
    df_conc.loc[2020:2024, gas] = pd.DataFrame((df_noaa_update.loc[2020:2024, gas]) - two_dataset_mean)

# %%
df_conc.tail(10)

# %%
# replot highlighting combined extended data
fig, ax = pl.subplots(4,4, figsize=(16, 16))
for igas, gas in enumerate(['HCFC-22', 'CFC-113', 'HCFC-141b', 'HCFC-142b', 'CH3CCl3', 'Halon-1211', 'Halon-1301', 'Halon-2402',
    'HFC-152a','HFC-143a', 'HFC-365mfc', 'HFC-227ea', 'HFC-23', 'SF6']):
    i = igas//4
    j = igas%4
    ax[i,j].plot(df_conc[gas], lw=3)
    ax[i,j].plot(df_noaa_update[gas])
    ax[i,j].plot(df_agage_ms[gas])
    ax[i,j].set_title(gas)
    ax[i,j].set_xlim(1980, 2025)

# %%
# CH3Br and HFC-32 combination of NOAA to 2023 and AGAGE MS

# %%
fig, ax = pl.subplots(1,2, figsize=(8, 4))
for igas, gas in enumerate(['CH3Br', 'HFC-32']):
    ax[igas].plot(df_conc[gas])
    ax[igas].plot(df_noaa[gas])
    ax[igas].plot(df_agage_ms[gas])
    ax[igas].set_title(gas)
    ax[igas].set_xlim(1980, 2025)

# in the HFC-32 case, take the mean of the overlapping period since the data has diverged a bit from AR6, and use a 
# trend extrapolation (last step) rather than a mean offset

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    gas = 'CH3Br'
    two_dataset_mean = pd.DataFrame((df_noaa[gas] - df_agage_ms[gas])).mean().values[0]
    df_conc.loc[2020:2022, gas] = pd.DataFrame((df_noaa.loc[2020:2022, gas], df_agage_ms.loc[2020:2022, gas])).mean()
    df_conc.loc[2023:2023, gas] = pd.DataFrame((df_noaa.loc[2023:2023, gas]) - two_dataset_mean)

    gas = 'HFC-32'
#    two_dataset_mean = pd.DataFrame((df_noaa[gas] - df_agage_ms[gas])).mean().values[0]
    df_conc.loc[2004:2022, gas] = pd.DataFrame((df_noaa.loc[2004:2022, gas], df_agage_ms.loc[2004:2022, gas])).mean()
#    df_conc.loc[2023:2023, gas] = pd.DataFrame((df_noaa.loc[2023:2023, gas]) - two_dataset_mean)

# %%
fig, ax = pl.subplots(1,2, figsize=(8, 4))
for igas, gas in enumerate(['CH3Br', 'HFC-32']):
    ax[igas].plot(df_conc[gas], lw=3)
    ax[igas].plot(df_noaa[gas])
    ax[igas].plot(df_agage_ms[gas])
    ax[igas].set_title(gas)
    ax[igas].set_xlim(1980, 2025)

# %%
df_agage_recent.columns

# %%
df_conc.columns

# %%
df_noaa.columns

# %%
df_noaa_update.columns

# %%
df_update.columns

# %%
df_agage_ms.columns

# %%
[col for col in df_noaa.columns if col not in df_noaa_update.columns]

# %%
df_noaa.tail()

# %%
df_conc.tail(10)

# %%
df_agage_ms.columns

# %%
df_update.rename(columns={'H-2402': 'Halon-2402'}, inplace=True)
df_update.rename(columns={'H-1211': 'Halon-1211'}, inplace=True)
df_update.rename(columns={'H-1301': 'Halon-1301'}, inplace=True)
df_update.rename(columns={'H-1202': 'Halon-1202'}, inplace=True)

# %%
df_agage_ms.rename(columns={'PFC-14': 'CF4'}, inplace=True)
df_agage_ms.rename(columns={'PFC-116': 'C2F6'}, inplace=True)
df_agage_ms.rename(columns={'PFC-218': 'C3F8'}, inplace=True)

# %%
df_agage_ms.rename(columns={'HFC-4310mee': 'HFC-43-10mee'}, inplace=True)

# %%
df_conc.columns

# %%
# where AGAGE MS data exists to 2022 but no corresponding value from other datasets, use this, with an offset from the mean overlap.
fig, ax = pl.subplots(4,4, figsize=(16, 16))
for igas, gas in enumerate(['HFC-236fa', 'HFC-245fa', 'HFC-43-10mee', 'CH3Cl', 'CH2Cl2', 'CHCl3',
    'SO2F2', 'NF3', 'CF4', 'C2F6', 'C3F8', 'CFC-13', 'CFC-114', 'CFC-115']):
    i = igas//4
    j = igas%4
    ax[i,j].plot(df_conc[gas])
    ax[i,j].plot(df_agage_ms[gas])
    if gas not in ['HFC-43-10mee']:
        ax[i,j].plot(df_update[gas])
    ax[i,j].set_title(gas)
    ax[i,j].set_xlim(1980, 2025)

# replace with AGAGE no offset, all available years :
# 236fa (after clean), 245fa (after clean), 4310mee (after clean), CH3Cl, SO2F2 (after clean), NF3 (after clean), CF4, C2F6, C3F8, CFC13, CFC115

# extend with AGAGE, no offset
# CHCl3

# extend with AGAGE, with offset
# CH2Cl2

# use 2023 update
# CFC114

# %%
df_conc.loc[2007:2022, 'HFC-236fa'] = df_agage_ms.loc[2007:2022, 'HFC-236fa']
df_conc.loc[2007:2022, 'HFC-245fa'] = df_agage_ms.loc[2007:2022, 'HFC-245fa']
df_conc.loc[2011:2022, 'HFC-43-10mee'] = df_agage_ms.loc[2011:2022, 'HFC-43-10mee']
df_conc.loc[2005:2022, 'SO2F2'] = df_agage_ms.loc[:, 'SO2F2']
df_conc.loc[2015:2022, 'NF3'] = df_agage_ms.loc[2015:2022, 'NF3']
df_conc.loc[2004:2022, 'CH3Cl'] = df_agage_ms.loc[2004:2022, 'CH3Cl']
df_conc.loc[2004:2022, 'CF4'] = df_agage_ms.loc[2004:2022, 'CF4']
df_conc.loc[2004:2022, 'C2F6'] = df_agage_ms.loc[2004:2022, 'C2F6']
df_conc.loc[2004:2022, 'C3F8'] = df_agage_ms.loc[2004:2022, 'C3F8']
df_conc.loc[2004:2022, 'CFC-13'] = df_agage_ms.loc[2004:2022, 'CFC-13']
df_conc.loc[2004:2022, 'CFC-115'] = df_agage_ms.loc[2004:2022, 'CFC-115']
df_conc.loc[2020:2022, 'CHCl3'] = df_agage_ms.loc[2020:2022, 'CHCl3']

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    gas = 'CH2Cl2'
    offset = pd.DataFrame((df_agage_ms[gas] - df_conc[gas])).mean().values[0]
    df_conc.loc[2020:2022, gas] = pd.DataFrame((df_agage_ms.loc[2020:2022, gas]) - offset)

# %%
df_conc.loc[2020:2023, 'CFC-114'] = df_update.loc[2020:2023, 'CFC-114']

# %%
# let's see how it looks post-update
fig, ax = pl.subplots(4,4, figsize=(16, 16))
for igas, gas in enumerate(['HFC-236fa', 'HFC-245fa', 'HFC-43-10mee', 'CH3Cl', 'CH2Cl2', 'CHCl3',
    'SO2F2', 'NF3', 'CF4', 'C2F6', 'C3F8', 'CFC-13', 'CFC-114', 'CFC-115']):
    i = igas//4
    j = igas%4
    ax[i,j].plot(df_conc[gas], lw=3)
    ax[i,j].plot(df_agage_ms[gas])
    if gas not in ['HFC-43-10mee']:
        ax[i,j].plot(df_update[gas])
    ax[i,j].set_title(gas)
    ax[i,j].set_xlim(1980, 2025)

# %%
# anything still left take from the 2023 update
df_conc.tail(11)

# %%
common_gases = [gas for gas in df_conc.columns if gas in df_update.columns]
df_conc.loc[2020, common_gases]

# %%
for gas in ['CFC-112', 'CFC-113a', 'HCFC-133a']:
    df_conc.loc[2019:2023, gas] = df_update.loc[2019:2023, gas]
    df_conc.loc[2015:2019, gas] = df_conc.loc[2015:2019, gas].interpolate()

# %%
df_conc.loc[1850:1989, 'i-C6F14'] = 0
df_conc.loc[1990:2015, 'i-C6F14'] = df_conc.loc[1990:2015, 'i-C6F14'].interpolate()#(inplace=True)

# %%
df_conc.loc[1850:1977, 'CFC-112'] = 0
df_conc.loc[1850:1977, 'CFC-112a'] = 0
df_conc.loc[1850:1977, 'CFC-113a'] = 0
df_conc.loc[1850:1977, 'CFC-114a'] = 0
df_conc.loc[1850:1979, 'HCFC-133a'] = 0
df_conc.loc[1850:1999, 'HCFC-31'] = 0
df_conc.loc[1850:2003, 'HCFC-124'] = 0


# %%
# Function to curve fit to the data
def linear(x, c, d):
    return c * x + d

# Initial parameter guess, just to kick off the optimization
guess = (1, 0)

# Place to store function parameters for each column
col_params = {}

# Curve fit each column
for col in df_conc.columns:
    # Create copy of data to remove NaNs for curve fitting
    fit_df = df_conc[col].dropna()

    # Get x & y
    x = fit_df.index.astype(float).values[-5:]
    y = fit_df.values[-5:]
    print (col, x, y)
    # Curve fit column and get curve parameters
    params = curve_fit(linear, x, y, guess)
    # Store optimized parameters
    col_params[col] = params[0]

# Extrapolate each column
for col in df_conc.columns:
    # Get the index values for NaNs in the column
    x = df_conc[pd.isnull(df_conc[col])].index.astype(float).values
    print(col, x)
    # Extrapolate those points with the fitted function
    df_conc.loc[x, col] = linear(x, *col_params[col])

# %%
df_conc.head()

# %%
df_conc.tail(11)

# %%
os.makedirs('../output', exist_ok = True)
df_conc.to_csv('../output/ghg_concentrations_1750-2024.csv')

# %% [markdown]
# ## Aggregated categories

# %%
gases_hfcs = [
    'HFC-134a',
    'HFC-23', 
    'HFC-32', 
    'HFC-125',
    'HFC-143a', 
    'HFC-152a', 
    'HFC-227ea', 
    'HFC-236fa', 
    'HFC-245fa', 
    'HFC-365mfc',
    'HFC-43-10mee',
]
gases_montreal = [
    'CFC-12',
    'CFC-11',
    'CFC-113',
    'CFC-114',
    'CFC-115',
    'CFC-13',
    'HCFC-22',
    'HCFC-141b',
    'HCFC-142b',
    'CH3CCl3',
    'CCl4',  # yes
    'CH3Cl',  # no
    'CH3Br',  # yes
    'CH2Cl2',  # no!
    'CHCl3',  # no
    'Halon-1211',
    'Halon-1301',
    'Halon-2402',
    'CFC-112',
    'CFC-112a',
    'CFC-113a',
    'CFC-114a',
    'HCFC-133a',
    'HCFC-31',
    'HCFC-124'
]
gases_pfc = [
    'CF4',
    'C2F6',
    'C3F8',
    'c-C4F8',
    'n-C4F10',
    'n-C5F12',
    'n-C6F14',
    'i-C6F14',
    'C7F16',
    'C8F18',
]

# %%
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

# %%
pfc_hfc134a_eq_1750 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_1750 = pfc_hfc134a_eq_1750 + (df_conc.loc[1750, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_1750 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_1750 = hfc_hfc134a_eq_1750 + (df_conc.loc[1750, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_1750 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_1750 = montreal_cfc12_eq_1750 + (df_conc.loc[1750, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_1750, hfc_hfc134a_eq_1750, montreal_cfc12_eq_1750

# %%
pfc_hfc134a_eq_1850 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_1850 = pfc_hfc134a_eq_1850 + (df_conc.loc[1850, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_1850 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_1850 = hfc_hfc134a_eq_1850 + (df_conc.loc[1850, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_1850 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_1850 = montreal_cfc12_eq_1850 + (df_conc.loc[1850, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_1850, hfc_hfc134a_eq_1850, montreal_cfc12_eq_1850

# %%
pfc_hfc134a_eq_2019 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2019 = pfc_hfc134a_eq_2019 + (df_conc.loc[2019, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2019 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2019 = hfc_hfc134a_eq_2019 + (df_conc.loc[2019, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2019 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2019 = montreal_cfc12_eq_2019 + (df_conc.loc[2019, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2019, hfc_hfc134a_eq_2019, montreal_cfc12_eq_2019

# %%
pfc_hfc134a_eq_2022 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2022 = pfc_hfc134a_eq_2022 + (df_conc.loc[2022, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2022 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2022 = hfc_hfc134a_eq_2022 + (df_conc.loc[2022, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2022 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2022 = montreal_cfc12_eq_2022 + (df_conc.loc[2022, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2022, hfc_hfc134a_eq_2022, montreal_cfc12_eq_2022

# %%
pfc_hfc134a_eq_2023 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2023 = pfc_hfc134a_eq_2023 + (df_conc.loc[2023, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2023 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2023 = hfc_hfc134a_eq_2023 + (df_conc.loc[2023, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2023 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2023 = montreal_cfc12_eq_2023 + (df_conc.loc[2023, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2023, hfc_hfc134a_eq_2023, montreal_cfc12_eq_2023

# %%
pfc_hfc134a_eq_2024 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2024 = pfc_hfc134a_eq_2024 + (df_conc.loc[2024, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2024 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2024 = hfc_hfc134a_eq_2024 + (df_conc.loc[2024, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2024 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2024 = montreal_cfc12_eq_2024 + (df_conc.loc[2024, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2024, hfc_hfc134a_eq_2024, montreal_cfc12_eq_2024

# %%
