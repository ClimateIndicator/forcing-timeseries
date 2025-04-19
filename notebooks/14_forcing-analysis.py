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
# # Forcing reporting, analysis and processing
#
# Take the raw ERF probablistic ensemble and produce output needed by the DAMIP folks.
#
# 1000 ensemble members of the 100,000:
# - all forcings
# - natural
# - well-mixed GHGs (does not include ozone or stratospheric water vapour)
# - other anthropogenic
#
# Stratospheric water vapour is not included in WMGHGs, as it appears the intention in DAMIP was to exclude it. From [Gillett et al. (2016)](https://gmd.copernicus.org/articles/9/3685/2016/gmd-9-3685-2016.pdf) (original emphasis):
#
# "**hist-GHG**:  These  historical  greenhouse-gas-only  simulations  resemble  the  historical  simulations  but  instead  are forced by *well-mixed* greenhouse gas changes only, similarly to the CMIP5 historicalGHG experiment. historical, hist-nat,and hist-GHG will allow the attribution of observed climate change to natural, greenhouse gas, and other anthropogenic components.  Models  with  interactive  chemistry  schemes should  either  turn  off  the  chemistry  or  use  a  preindustrial climatology of stratospheric and tropospheric ozone in their radiation  schemes."
#
# Stratospheric water vapour from methane oxidation would only be produced in models with interactive chemistry, therefore the intention appears to be to exclude it from the definition of WMGHG.

# %%
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import xarray as xr
import json

# %%
ds = xr.load_dataset('../output/ERF_ensemble.nc')

# %%
variables = list(ds.keys())

# %%
df = pd.read_csv('../output/ERF_best_1750-2024.csv', index_col=0)

# %%
SAMPLES = 100000

# %% [markdown]
# ## Non-aggregated statistics
#
# 2011, 2019 and 2024

# %%
for variable in [
    'CO2', 
    'CH4', 
    'N2O', 
    'aerosol-radiation_interactions', 
    'aerosol-cloud_interactions', 
    'O3',
    'contrails',
    'land_use',
    'BC_on_snow',
    'H2O_stratospheric',
    'solar'
]:
    print(variable, np.percentile(ds[variable].loc[dict(time=2024)], (5, 50, 95)), df.loc[2024, variable])

# %% [markdown]
# ## Aggregated categories

# %%
total_best     = np.zeros((275))
natural_best   = np.zeros((275))
aerosol_best   = np.zeros((275))
wmghg_best     = np.zeros((275))
other_ant_best = np.zeros((275))
halogen_best   = np.zeros((275))

for variable in tqdm(variables):
    #print(variable)
    total_best = total_best + df[variable]
    if variable in ['solar', 'volcanic']:
        natural_best = natural_best + df[variable]
    elif variable not in [
        'aerosol-radiation_interactions', 'aerosol-cloud_interactions',
        'O3', 'H2O_stratospheric', 'contrails', 'BC_on_snow', 'land_use'
    ]:
        wmghg_best = wmghg_best + df[variable]
    else:
        other_ant_best = other_ant_best + df[variable]
    if variable not in ['solar', 'volcanic', 'aerosol-radiation_interactions', 'aerosol-cloud_interactions',
        'O3', 'H2O_stratospheric', 'contrails', 'BC_on_snow', 'land_use', 'CO2', 'CH4', 'N2O']:
        halogen_best = halogen_best + df[variable]
    if variable in ['aerosol-radiation_interactions', 'aerosol-cloud_interactions']:
        aerosol_best = aerosol_best + df[variable]
type(total_best)

# %%
df_out = pd.concat(
    [
        total_best, natural_best, total_best-natural_best, wmghg_best, other_ant_best
    ], axis=1
)
df_out.columns = ['total', 'natural', 'anthropogenic', 'wmghgs', 'other_ant']
df_out.to_csv('../output/ERF_best_DAMIP_1750-2024.csv')

# %%
total     = np.zeros((275, SAMPLES))
natural   = np.zeros((275, SAMPLES))
aerosol   = np.zeros((275, SAMPLES))
wmghg     = np.zeros((275, SAMPLES))
other_ant = np.zeros((275, SAMPLES))
halogen   = np.zeros((275, SAMPLES))

for variable in tqdm(variables):
    #print(variable)
    total = total + ds[variable]
    if variable in ['solar', 'volcanic']:
        natural = natural + ds[variable]
    elif variable not in [
        'aerosol-radiation_interactions', 'aerosol-cloud_interactions',
        'O3', 'H2O_stratospheric', 'contrails', 'BC_on_snow', 'land_use'
    ]:
        wmghg = wmghg + ds[variable]
    else:
        other_ant = other_ant + ds[variable]
    if variable not in ['solar', 'volcanic', 'aerosol-radiation_interactions', 'aerosol-cloud_interactions',
        'O3', 'H2O_stratospheric', 'contrails', 'BC_on_snow', 'land_use', 'CO2', 'CH4', 'N2O']:
        halogen = halogen + ds[variable]
    if variable in ['aerosol-radiation_interactions', 'aerosol-cloud_interactions']:
        aerosol = aerosol + ds[variable]

# %% [markdown]
# ## Show shape matters

# %%
pl.plot(np.arange(1750, 2025), aerosol[:, :7]);

# %%
print('halogen', np.percentile(halogen[-1,:], (5, 50, 95)), halogen_best[2024])

# %%
# total     = np.zeros((273, 100000))
# natural   = np.zeros((273, 100000))
# aerosol   = np.zeros((273, 100000))
# wmghg     = np.zeros((273, 100000))
# other_ant = np.zeros((273, 100000)) 

print('total:        ', np.percentile(total[-1, :], (5, 50, 95)))
print('anthropogenic:', np.percentile(total[-1, :]-natural[-1, :], (5, 50, 95)))
print('natural:      ', np.percentile(natural[-1, :], (5, 50, 95)))
print('aerosol:      ', np.percentile(aerosol[-1, :], (5, 50, 95)))
print('wmghg:        ', np.percentile(wmghg[-1, :], (5, 50, 95)))
print('other_ant:    ', np.percentile(other_ant[-1, :], (5, 50, 95)))

# %%
# pick 1000 for export
with open('../data/random_seeds.json') as f:
    seeds = json.load(f)

# %%
np.random.seed(seeds[99])
subset = np.random.choice(np.arange(SAMPLES), 1000, replace=False)

# %%
print('total:        ', np.percentile(total[-1, subset], (5, 50, 95)))
print('anthropogenic:', np.percentile(total[-1, subset]-natural[-1, subset], (5, 50, 95)))
print('natural:      ', np.percentile(natural[-1, subset], (5, 50, 95)))
print('aerosol:      ', np.percentile(aerosol[-1, subset], (5, 50, 95)))
print('wmghg:        ', np.percentile(wmghg[-1, subset], (5, 50, 95)))
print('other_ant:    ', np.percentile(other_ant[-1, subset], (5, 50, 95)))

# %%
xr.Dataset(
    {
        'total': total[:, subset],
        'natural': natural[:, subset],
        'wmghg': wmghg[:, subset],
        'other_ant': other_ant[:, subset]
    }
).to_netcdf('../output/ERF_DAMIP_1000.nc')

# %%
ds.close()

# %%
