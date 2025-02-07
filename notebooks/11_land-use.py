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
# # Land use forcing
#
# Land use albedo: new data from Thomas Gasser based on HYDE-LUH
#
# Irrigation: from Chris Wells, based on FAO water withdrawal data, scaled to IPCC AR6 assessment for 2019 based on Sherwood et al (2018)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.stats import linregress

# %%
df_landuse = pd.read_csv('../data/land_use/SARF_LCC_Ghimire2014+Ouyang2022+GCB2024.csv', index_col=0)

# %%
pl.plot(df_landuse)

# %%
# Take Thomas' suggestion of scaling the LUH1 data to the IPCC assessment, then using this scaling factor to adjust the LUH2/GCB time series.
# actually no since we don't have 2005 in this dataset - let's scale LUH2 to AR6

# %%
df_landuse.columns

# %%
df_landuse.loc[2019, 'LUH2-GCB2024']

# %%
pl.plot(-0.15 / df_landuse.loc[2005, 'LUH2-GCB2024'] * df_landuse['LUH2-GCB2024'])

# %%
df_landuse['LUH2-GCB2024_rescaled'] = -0.15 / df_landuse.loc[2004, 'LUH2-GCB2024'] * df_landuse['LUH2-GCB2024']

# %%
# estimate 2024 based on persistence
df_landuse

# %%
df_landuse.loc[2024, 'LUH2-GCB2024_rescaled'] = df_landuse.loc[2023, 'LUH2-GCB2024_rescaled']

# %%
df_landuse

# %%
pl.plot(df_landuse)

# %%
df_irr = pd.read_csv('../data/land_use/Irrigation_ERF_Wells_FAO.csv', index_col=0)
df_irr

# %%
lr = linregress(df_irr.loc[2013:].index, df_irr.loc[2013:, "Irrigation ERF"])

# %%
pl.plot(df_irr.loc[2013:])
pl.plot(np.arange(2013, 2025), lr.slope * np.arange(2013, 2025) + lr.intercept)

# %%
df_irr.loc[2023, "Irrigation ERF"] = lr.slope * 2023 + lr.intercept
df_irr.loc[2024, "Irrigation ERF"] = lr.slope * 2024 + lr.intercept

# %%
pl.plot(df_irr)

# %%
df_landuse.loc[1750:2024, 'irrigation'] = df_irr.values

# %%
df_landuse['total'] = df_landuse['LUH2-GCB2024_rescaled'] + df_landuse['irrigation']

# %%
df_landuse

# %%
pl.plot(df_landuse['total'])

# %%
df_landuse.loc[1750:, ['LUH2-GCB2024_rescaled', 'irrigation', 'total']].to_csv('../output/land_use_1750-2024.csv')

# %%
df_landuse.loc[1750:, ['LUH2-GCB2024_rescaled', 'irrigation', 'total']]

# %%
