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
# # Temperature plot
#
# Make the figure of the GMST anomalies

# %%
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

import os

# %%
os.makedirs('../plots', exist_ok=True)

# %%
df_gmst = pd.read_csv('../data/gmst/IGCC_GMST_1850-2024.csv')

# %%
df_gmst

# %%
#df_gmst.index#.rolling(10).mean()

# %%
df_gmst_10=df_gmst.rolling(10).mean()

# %%
pl.rcParams['figure.figsize'] = (12/2.54, 9/2.54)
pl.rcParams['font.size'] = 11
pl.rcParams['font.family'] = 'Arial'
pl.rcParams['xtick.direction'] = 'in'
pl.rcParams['xtick.minor.visible'] = True
pl.rcParams['xtick.top'] = True
pl.rcParams['ytick.direction'] = 'in'
pl.rcParams['ytick.right'] = True
pl.rcParams['ytick.minor.visible'] = True
pl.rcParams['xtick.top'] = True
pl.rcParams['figure.dpi'] = 150

# %%
pl.plot(df_gmst.year, df_gmst.gmst, lw=1, color='r')
pl.plot(df_gmst_10.year, df_gmst_10.gmst, lw=2.5, color='k')
pl.xlim(1850, 2025)
pl.ylabel('Temperature anomaly\nrelative to 1850-1900 (°C)')
pl.tight_layout()
pl.savefig('../plots/gmst.png')
pl.savefig('../plots/gmst.pdf')

# %%
