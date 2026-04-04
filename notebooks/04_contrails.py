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
# # Contrails and contrail cirrus
#
# Basic method follows [Lee et al. (2020)](https://www.sciencedirect.com/science/article/pii/S1352231020305689). We extend beyond the end of the Lee assessment (2018) by using:
#
# - Lee et al. (2020); there is a supplementary Excel dataset that details all of the calculations, including the conversion between distance travelled and contrail radiative forcing (a constant value); the nub is therefore extracting the distance travelled from post-2018 data. We have fule consumption estimates as follows:
#
# - Sausen and Schumann (2000) for the 1940-1970 period.
#   
# - IEA World Oil Statistics, Edition 2024, for World region and the sum of DAVGAS, DJETGAS and DJETKERO. We use data for 1971-2022. This is paywalled data so we do not include it in the repository. It is from https://www.iea.org/data-and-statistics/data-product/oil-information#world-oil-statistics. It is put into a CSV form for reading into this repository.
#
# - IATA (2024), aviation fuel use, table 7 of https://www.iata.org/en/iata-repository/publications/economic-reports/global-outlook-for-air-transport-december-2024/. This data is available to 2024 as an estimate and 2025 as a projection. It could fully replace the IEA data, but we continue to use IEA to follow Lee. Annoyingly, they decide to switch from litres to gallons in 2024 (why??? nobody uses imperial units now!)

# %% [markdown]
# In Lee et al. (2020), contrail ERF is calculated as:
#
# distance (km) * scale factor (1.17) * contrail forcing per km (see below)
#
# The scale factor converts horizontal distance travelled to 3D distance.
#
# Lee et al assume a constant fuel consumption (CO2 emission per fuel burn) of 3.16 kgCO2 per kg fuel
#
# From Wiki (https://en.wikipedia.org/wiki/Aviation_fuel) I get 0.81 kg/L (accessed 2024-02-27)
#
# RPKs, ASKs and distances from https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/ for 1950-2022

# %% [markdown]
# ## recipe
# 1. read in Lee spreadsheet
# 2. derive contrails per kilogram fuel burn
# 3. calculate IEA factor offline and apply - read in this time series
# 4. apply factor to IATA data
# 5. check overlaps and construct time series

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

# %%
kg_co2_per_kg_fuel = 3.16

# %%
us_gallons_to_litres = 3.78541

# %%
iata = pd.DataFrame(np.array([96, 52, 62, 76, 92, 101, 107]) * 1e9 * us_gallons_to_litres, index=pd.Index(np.arange(2019, 2026), name='year'), columns=['Aviation fuel, litres'])

# %%
iata

# %%
# according to Wiki, aviation fuel has a density of around 0.81 kg/L, so convert to kilograms
kg_per_litre = 0.81
iata['Aviation fuel, kg'] = iata['Aviation fuel, litres'] * kg_per_litre
iata['Aviation emissions, MtCO2/yr'] = iata['Aviation fuel, kg'] * kg_co2_per_kg_fuel / 1e9
iata

# %%
iea = pd.read_csv('../data/contrails/OIWORLD-2024.csv', index_col="year")
iea['Aviation fuel, kg'] = iea['Aviation fuel, kilotons'] * 1e6
iea['Aviation emissions, MtCO2/yr'] = iea['Aviation fuel, kg'] * kg_co2_per_kg_fuel / 1e9
iea

# %%
sausen_schumann = pd.read_csv('../data/contrails/sausen_schumann_co2.csv', index_col=0)

# %%
sausen_schumann

# %%
# make a column in CO2 units, which is comparible directly to Lee et al.
sausen_schumann['Aviation emissions, MtCO2/yr'] = sausen_schumann['aviation_emissions_teragrams_carbon'] * 3.664
sausen_schumann

# %%
lee2020 = pd.read_csv('../data/contrails/lee2020_clean_data.csv', index_col=0)
lee2020.rename(columns={'CO2 Tg/yr': 'Aviation emissions, MtCO2/yr'}, inplace=True)

# %%
pl.plot(sausen_schumann['Aviation emissions, MtCO2/yr'])
pl.plot(iea['Aviation emissions, MtCO2/yr'])
pl.plot(lee2020['Aviation emissions, MtCO2/yr'])
pl.plot(iata['Aviation emissions, MtCO2/yr'])

# %%
constructed_emissions = pd.DataFrame(np.ones(95)*np.nan, index=pd.Index(np.arange(1930, 2025), name='year'), columns=['MtCO2/yr'])

# %%
constructed_emissions.loc[1940:1970, 'MtCO2/yr'] = sausen_schumann.loc[1940:1970, 'Aviation emissions, MtCO2/yr']
constructed_emissions.loc[1971:2022, 'MtCO2/yr'] = iea.loc[1971:2022, 'Aviation emissions, MtCO2/yr']
constructed_emissions.loc[2023:2024, 'MtCO2/yr'] = iata.loc[2023:2024, 'Aviation emissions, MtCO2/yr']
constructed_emissions.loc[1930, 'MtCO2/yr'] = 0
constructed_emissions.interpolate(inplace=True)

# %%
pl.plot(constructed_emissions)

# %%
pl.plot(lee2020['Scaled distance million km']/lee2020['Aviation emissions, MtCO2/yr']/ 1000)
pl.title('Fuel efficiency')
pl.ylabel('km / kgCO2')

# %%
constructed_efficiency = pd.DataFrame(np.ones(95) * np.nan, index=pd.Index(np.arange(1930, 2025), name='year'), columns=['km / kgCO2'])
constructed_efficiency.loc[1990:2018, 'km / kgCO2'] = lee2020['Scaled distance million km']/lee2020['Aviation emissions, MtCO2/yr']/ 1000
#constructed_efficiency[2019] = 
# For 2019 to 2022, take estimates of distance from https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/
# scale up by 1.17 as in Lee
constructed_efficiency.loc[2019, 'km / kgCO2'] = 1.17 * 56199 / iea.loc[2019, 'Aviation emissions, MtCO2/yr'] / 1000
constructed_efficiency.loc[2020, 'km / kgCO2'] = 1.17 * 28013 / iea.loc[2020, 'Aviation emissions, MtCO2/yr'] / 1000
constructed_efficiency.loc[2021, 'km / kgCO2'] = 1.17 * 33705 / iea.loc[2021, 'Aviation emissions, MtCO2/yr'] / 1000
constructed_efficiency.loc[2022, 'km / kgCO2'] = 1.17 * 42801 / iea.loc[2022, 'Aviation emissions, MtCO2/yr'] / 1000

# for 2023 and 2024, we have no better means than using 2022 data. But really, we'd expect efficiency to improve, since there were probably
# fewer half-empty planes after COVID.
constructed_efficiency.loc[2023:2024, 'km / kgCO2'] = constructed_efficiency.loc[2022, 'km / kgCO2']

# %% [markdown]
# Now try to port this backwards
#
# Lee et al: "Aviation transport efficiency has improved by approximately eightfold since 1960". I know this is based on RPKs rather than actual kilometers.
#
# According to https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/, capacity increased from about 60% in 1960 to about 80% in 2018/2019.
#
# So 1/3 of the improvement is due to increasing capacity, and 2/3 * 8 is due to aircraft efficiency.

# %%
constructed_efficiency.loc[1990, 'km / kgCO2'] / constructed_efficiency.loc[2018, 'km / kgCO2']

# %%
constructed_efficiency.loc[1960, 'km / kgCO2'] = constructed_efficiency.loc[2018, 'km / kgCO2'] / (8 * 2/3)
constructed_efficiency.loc[1930, 'km / kgCO2'] = constructed_efficiency.loc[1960, 'km / kgCO2']
#constructed_efficiency.loc[1989, 'km / kgCO2'] = constructed_efficiency.loc[2018, 'km / kgCO2'] * 9/16
constructed_efficiency.interpolate(inplace=True)

# %%
pl.plot(constructed_efficiency)

# %%
constructed_efficiency

# %%
constructed_scaled_distance = pd.DataFrame(np.ones(95) * np.nan, index=pd.Index(np.arange(1930, 2025), name='year'), columns=['Scaled distance, km'])
constructed_scaled_distance.loc[1930:2024, 'Scaled distance, km'] = (constructed_emissions.values * constructed_efficiency.values).squeeze() * 1e9
pl.plot(constructed_scaled_distance)

# %%
lee_conversion_km_to_contrail_erf = 9.3595037832885E-13  # cell J73 of supplementary Excel, divide by 1000 to convert milliwatts to watts

# %%
# Do we keep the 2018 value in Lee (0.0574) by scaling this, or use the slightly newer data from IEA? TBF it is the same to 3 d.p.
constructed_scaled_distance.loc[2018, 'Scaled distance, km'] * lee_conversion_km_to_contrail_erf

# %%
# contrail forcing ERF
contrails_erf = pd.DataFrame(np.ones(95) * np.nan, index=pd.Index(np.arange(1930, 2025), name='year'), columns=['Contrails ERF, W/m2'])
contrails_erf.loc[1930:2024, 'Contrails ERF, W/m2'] = (constructed_scaled_distance * lee_conversion_km_to_contrail_erf).values.squeeze()

pl.plot(contrails_erf)

# %%
contrails_erf.to_csv('../output/contrails_ERF_1930-2024.csv')

# %%
