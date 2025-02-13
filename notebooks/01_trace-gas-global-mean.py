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
# # Calculate GHG concentrations
#
# **update Jan 2025: while we wait for Brad and Lindsay's assessments, we will extrapolate last year's numbers for the preliminary WMO data.**
#
# **update Feb 2025: preliminary 2024 values from Lindsay in the paper draft, use these here for WMO analysis**
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
# NOAA (accessed 2025-02-12):
# - https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt
# - https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_annmean_gl.txt
# - https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_annmean_gl.txt
# - https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_annmean_gl.txt
# - https://gml.noaa.gov/aftp/data/hats/Total_Cl_Br/2024%20update%20total%20Cl%20Br%20&%20F.xls  (converted to CSV with header and footer rows stripped out; save as noaa_**YYYY**_global_mean_mixing_ratios.csv) **note: each year, check the FTP directory to see if there has been an annual update**
#
# AGAGE (accessed 2025-02-12):
# - https://agage2.eas.gatech.edu/data_archive/global_mean/global_mean_ms.txt
# - https://agage2.eas.gatech.edu/data_archive/global_mean/global_mean_md.txt
#
# One final update is to update 2023 to incorporate CSIRO measurements for CO2, CH4 and N2O. These values from Brad.

# %%
import os
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

# %%
df_noaa

# %%
df_noaa.loc[162:167,'CFC-12']

# %%
df_noaa[df_noaa=="ND"]=np.nan

# %%
df_noaa = df_noaa.rolling(6, center=True).mean()
df_noaa['YYYY'] = df_noaa.date-0.5
df_noaa.drop(df_noaa.tail(2).index,inplace=True)
df_noaa.drop(df_noaa.head(3).index,inplace=True)
df_noaa.set_index('YYYY', inplace=True)
df_noaa.drop(columns=['date'], inplace=True)
df_noaa.rename(columns={'H2402': 'H-2402'}, inplace=True)
df_noaa = df_noaa[df_noaa.index % 1 == 0]
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
df_co2

# %%
# this comes from Brad and Lindsay
df_update = pd.read_csv(
    '../data/ghg_concentrations/ar6_updated/update_2019-2023.csv',
    index_col=0
)

# note CO2 is on X2007 scale in the 2022 Climate Indicators - the offset is 0.18 ppm
df_update

# %%
# typically year of assessment minus 1
df_ch4_noaa.loc[1984:2023, 'mean']

# %%
# difference between NOAA and NOAA/AGAGE is small
# last two ranges typically year of assessment minus one
pd.concat((df_conc.loc[1984:2019, 'CH4'], df_update.loc[2020:2023, 'CH4']), axis=0) - df_ch4_noaa.loc[1984:2023, 'mean']

# %%
# end is year of assessment minus one
df_n2o_noaa.loc[2001:2023, 'mean']

# %%
# For N2O it's a bit bigger. Lindsay recommends scaling NOAA up by 1.0007
# again typically assessment year minus one
pd.concat((df_conc.loc[2001:2019, 'N2O'], df_update.loc[2020:2023, 'N2O']), axis=0) - df_n2o_noaa.loc[2001:2023, 'mean']

# %%
# as above
pd.concat((df_conc.loc[2001:2019, 'N2O'], df_update.loc[2020:2023, 'N2O']), axis=0) - 1.0007 * df_n2o_noaa.loc[2001:2023, 'mean']

# %%
# I do think that CH4 and N2O should include AGAGE

# CO2: this is on X2019 scale. Meinshausen et al (2017) is the source of pre-1980 data in AR6. 
# Meinshausen data should be adjusted for the X2019 scale.
# X2019 = 1.00079*X2007 - 0.142 (from Brad)
df_conc.loc[1750:1978, 'CO2'] = df_conc.loc[1750:1978, 'CO2'] * 1.00079 - 0.142

df_conc.loc[1979:2023, 'CO2'] = df_co2.loc[1979:2023, 'mean']

# For methane and N2O, the calibration scales have not changed, and we use multiple datasets, so continue with 2023 Indicators estimate and
# adjust 2023 NOAA-only value for the average of the differences between 2023 Indicators and NOAA
df_conc.loc[2020:2023, 'CH4':'N2O'] = df_update.loc[2020:2023, 'CH4':'N2O']

# %%
df_conc.columns

# %%
df_update.columns

# %%
df_update.rename(columns={'H-2402': 'Halon-2402'}, inplace=True)
df_update.rename(columns={'H-1211': 'Halon-1211'}, inplace=True)
df_update.rename(columns={'H-1301': 'Halon-1301'}, inplace=True)
df_update.rename(columns={'H-1202': 'Halon-1202'}, inplace=True)

# %%
# now let's incorporate Brad and Lindsay's new data
for species in df_update.columns:
    if species in ['Halon-1202']:
        continue
    df_conc.loc[2019:2023, species] = df_update.loc[2019:2023, species]
    
    
df_conc.loc[2024, 'CO2'] = 422.6
df_conc.loc[2024, 'CH4'] = 1930.9
df_conc.loc[2024, 'N2O'] = 337.8

# %%
df_conc

# %%
df_agage_md = pd.read_csv(
    '../data/ghg_concentrations/agage/global_mean_md.txt', 
    sep=r'\s+',
    skiprows=14,
    index_col=0
)

# %%
df_agage_md = df_agage_md.rolling(12, min_periods=12, center=True, step=12).mean().drop([col for col in df_agage_md.columns if '---' in col],axis=1)
df_agage_md.drop(columns='MM', inplace=True)
df_agage_md.set_index('YYYY', inplace=True)
df_agage_md.drop(index=np.nan, inplace=True)

# %%
df_agage_md

# %%
#df_agage_ms.loc[2008:2020, 'HFC4310mee']

# %%
#df_conc.loc[2011:2020, 'HFC-43-10mee'] = df_agage_ms.loc[2011:2020, 'HFC4310mee']

# %%
df_conc.loc[1850:1989, 'i-C6F14'] = 0
df_conc.loc[1990:2015, 'i-C6F14'].interpolate(inplace=True)

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
    df_conc[col][x] = linear(x, *col_params[col])

# %%
df_conc

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
