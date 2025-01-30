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
# # Calculate equivalent emissions from concentration time series

# %%
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from fair import FAIR, __version__
from fair.interface import fill, initialise

# %%
# leverage fair's concentrations to emissions routines and include all minor species in concentration time series
f = FAIR(temperature_prescribed=True)
f.define_time(1750, 2024, 1)
f.define_scenarios(["historical"])
f.define_configs(["historical"])
species = [
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
    'NF3',
    'SF6',
    'SO2F2',
    'CF4',
    'C2F6',
    'C3F8',
    'c-C4F8',
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
    'CCl4',
    'CH3Cl',
    'CH3Br',
    'CH2Cl2',
    'CHCl3',
    'Halon-1211',
    'Halon-1301',
    'Halon-2402',
    'n-C4F10',
    'n-C5F12',
    'n-C6F14',
    'i-C6F14',
    'C7F16',
    'C8F18',
    'CFC-112',
    'CFC-112a',
    'CFC-113a',
    'CFC-114a',
    'HCFC-133a',
    'HCFC-31',
    'HCFC-124'
]

# %%
# since we only care about back-calculated emissions and not interactions or
# climate effects, treat everything as an F-gas which is inert
properties = {
    specie: {
        "type": "f-gas",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    }
    for specie in species
}

# %%
f.define_species(species, properties)
f.allocate()

# %%
df_conc_obs=pd.read_csv('../output/ghg_concentrations_1750-2024.csv', index_col=0)

# %%
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

# %%
for specie in species:
    f.concentration.loc[
        dict(
            timebounds=slice(1751, 2025),
            specie=specie,
            scenario="historical",
            config="historical",
        )
    ] = 0.5 * (
        df_conc_obs.loc[1750:2023, specie].values
        + df_conc_obs.loc[1751:2024, specie].values
    )
    f.concentration.loc[
        dict(
            timebounds=1750,
            specie=specie,
            scenario="historical",
            config="historical",
        )
    ] = df_conc_obs.loc[1750, specie]

# %%
mol_wt = {
    'HFC-134a': 102.03, 
    'HFC-23': 78.014,
    'HFC-32': 52.023,
    'HFC-125': 120.02,
    'HFC-143a': 84.04,
    'HFC-152a': 66.05,
    'HFC-227ea': 170.03,
    'HFC-236fa': 152.04,
    'HFC-245fa': 134.05,
    'HFC-365mfc': 148.07,
    'HFC-43-10mee': 252.05,
    'NF3': 71.002,
    'SF6': 146.06,
    'SO2F2': 102.06,
    'CF4': 88.004,
    'C2F6': 138.01,
    'C3F8': 188.02,
    'c-C4F8': 200.03,
    'CFC-12': 120.91,
    'CFC-11': 137.36,
    'CFC-113': 187.37,
    'CFC-114': 170.92,
    'CFC-115': 154.46,
    'CFC-13': 104.46,
    'HCFC-22': 86.47,
    'HCFC-141b': 116.95,
    'HCFC-142b': 100.49,
    'CH3CCl3': 133.4,
    'CCl4': 153.8,
    'CH3Cl': 50.49,
    'CH3Br': 94.94,
    'CH2Cl2': 84.93,
    'CHCl3': 119.37,
    'Halon-1211': 165.36,
    'Halon-1301': 148.91,
    'Halon-2402': 259.82,
    'n-C4F10': 238.03,
    'n-C5F12': 288.03,
    'n-C6F14': 338.04,
    'i-C6F14': 338.04,
    'C7F16': 388.05,
    'C8F18': 438.06,
    'CFC-112': 203.82,
    'CFC-112a': 203.82,
    'CFC-113a': 187.37,
    'CFC-114a': 170.92,
    'HCFC-133a': 118.48,
    'HCFC-31': 68.48,
    'HCFC-124': 136.47 
}

# %%
lifetime = {
    'HFC-134a': 14, 
    'HFC-23': 228,
    'HFC-32': 5.4,
    'HFC-125': 30,
    'HFC-143a': 51,
    'HFC-152a': 1.6,
    'HFC-227ea': 36,
    'HFC-236fa': 213,
    'HFC-245fa': 7.9,
    'HFC-365mfc': 8.9,
    'HFC-43-10mee': 17,
    'NF3': 569,
    'SF6': 3200,
    'SO2F2': 36,
    'CF4': 50000,
    'C2F6': 10000,
    'C3F8': 2600,
    'c-C4F8': 3200,
    'CFC-12': 102,
    'CFC-11': 52,
    'CFC-113': 93,
    'CFC-114': 189,
    'CFC-115': 540,
    'CFC-13': 640,
    'HCFC-22': 11.9,
    'HCFC-141b': 9.4,
    'HCFC-142b': 18,
    'CH3CCl3': 5,
    'CCl4': 32,
    'CH3Cl': 0.9,
    'CH3Br': 0.8,
    'CH2Cl2': 0.493,
    'CHCl3': 0.501,
    'Halon-1211': 16,
    'Halon-1301': 72,
    'Halon-2402': 28,
    'n-C4F10': 2600,
    'n-C5F12': 4100,
    'n-C6F14': 3100,
    'i-C6F14': 3100,
    'C7F16': 3000,
    'C8F18': 3000,
    'CFC-112': 63.6,
    'CFC-112a': 52,
    'CFC-113a': 55,
    'CFC-114a': 105,
    'HCFC-133a': 55,
    'HCFC-31': 1.2,
    'HCFC-124': 5.9
}

# %%
for specie in species:
    f.species_configs['molecular_weight'].loc[dict(specie=specie)] = mol_wt[specie]
    f.species_configs['unperturbed_lifetime'].loc[dict(specie=specie)] = np.ones(4) * lifetime[specie]
    f.species_configs['partition_fraction'].loc[dict(specie=specie)] = [1, 0, 0, 0]

# %%
m = 1 / (5.1352 * f.species_configs["molecular_weight"] / 28.97)
c1 = f.concentration[0, ...]

# %%
m

# %%
for specie in species:
    fill(f.species_configs["baseline_concentration"], 0, specie=specie)
    fill(f.species_configs["baseline_emissions"], 0, specie=specie)
    fill(f.species_configs['forcing_reference_concentration'], 0, specie=specie)
    fill(f.species_configs['iirf_airborne'], 0, specie=specie)
    fill(f.species_configs['iirf_temperature'], 0, specie=specie)
    fill(f.species_configs['iirf_uptake'], 0, specie=specie)
    c1 = f.concentration.loc[
        dict(
            specie=specie,
            timebounds=1750,
            scenario="historical",
            config="historical",
        )
    ]
    m = 1 / (
        5.1352 * f.species_configs["molecular_weight"].loc[dict(specie=specie)] / 28.97
    )
    initialise(f.airborne_emissions, c1 / m, specie=specie)
    initialise(f.gas_partitions, np.array([c1 / m, 0, 0, 0]), specie=specie)

# %%
# don't calculate warming; we have to initialise it otherwise FaIR will complain about
# NaNs
fill(f.temperature, 0)

# %%
f.calculate_concentration_per_emission()

# %%
f.calculate_g()

# %%
f.calculate_iirf0()

# %%
f.run(progress=False)

# %%
f.species_configs

# %%
f.emissions.squeeze().to_pandas().to_csv('../output/ghg_equivalent_emissions.csv')

# %%
