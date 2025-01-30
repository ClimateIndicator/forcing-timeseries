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
# # Process MERRA2 reanalysis for aerosol loadings
#
# Aerosol loadings haven't changed much for SO2 and SO4 over the last 6 years, which is surprising (limited impact of COVID and IMO2020) whereas there was a noticable increase in BC and OC emissions in 2023 and 2024, probably related to biomass burning.
#
# The only real conclusion I can make is no conclusion - aerosol emissions did not change much from 2022 to 2024, since there's no good evidence to counter this.

# %%
import iris
import iris.coord_categorisation as cat
from iris.util import equalise_attributes, unify_time_units
import numpy as np

# %%
so2_cube = iris.load_cube(
    '../data/slcf_emissions/merra/SO2CMASS.nc'
)
so4_cube = iris.load_cube(
    '../data/slcf_emissions/merra/SO4CMASS.nc'
)
bc_cube = iris.load_cube(
    '../data/slcf_emissions/merra/BCCMASS.nc'
)
oc_cube = iris.load_cube(
    '../data/slcf_emissions/merra/OCCMASS.nc'
)

# %%
so2_cube

# %%
# unify_time_units(so2_cubes)
# unify_time_units(so4_cubes)

# %%
# equalise_attributes(so2_cubes)
# equalise_attributes(so4_cubes)

# %%
# # Remove all the other attributes.
# for cube in so2_cubes:
#     coord = cube.coord('time')
#     del coord.attributes["begin_date"]
#     del coord.attributes["begin_time"]
#     del coord.attributes["time_increment"]
#     del coord.attributes["valid_range"]
#     del coord.attributes["vmax"]
#     del coord.attributes["vmin"]

# for cube in so4_cubes:
#     coord = cube.coord('time')
#     del coord.attributes["begin_date"]
#     del coord.attributes["begin_time"]
#     del coord.attributes["time_increment"]
#     del coord.attributes["valid_range"]
#     del coord.attributes["vmax"]
#     del coord.attributes["vmin"]

# %%
# so2_cube = so2_cubes.concatenate_cube()
# so4_cube = so4_cubes.concatenate_cube()

# %%
# no bounds included so guess them
so2_cube.coord('latitude').guess_bounds()
so2_cube.coord('longitude').guess_bounds()
so4_cube.coord('latitude').guess_bounds()
so4_cube.coord('longitude').guess_bounds()
bc_cube.coord('latitude').guess_bounds()
bc_cube.coord('longitude').guess_bounds()
oc_cube.coord('latitude').guess_bounds()
oc_cube.coord('longitude').guess_bounds()

# %%
so2_area_weights = iris.analysis.cartography.area_weights(so2_cube)
so4_area_weights = iris.analysis.cartography.area_weights(so4_cube)
bc_area_weights = iris.analysis.cartography.area_weights(bc_cube)
oc_area_weights = iris.analysis.cartography.area_weights(oc_cube)

# %%
cat.add_year(so2_cube, 'time', name='year')
cat.add_year(so4_cube, 'time', name='year')
cat.add_year(bc_cube, 'time', name='year')
cat.add_year(oc_cube, 'time', name='year')
so2_cube_area_sum = so2_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=so2_area_weights) # kg
so4_cube_area_sum = so4_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=so4_area_weights) # kg
bc_cube_area_sum = bc_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=bc_area_weights) # kg
oc_cube_area_sum = oc_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=oc_area_weights) # kg
so2_cube_global_annual_mean = so2_cube_area_sum.aggregated_by('year', iris.analysis.MEAN) # kg
so4_cube_global_annual_mean = so4_cube_area_sum.aggregated_by('year', iris.analysis.MEAN) # kg
bc_cube_global_annual_mean = bc_cube_area_sum.aggregated_by('year', iris.analysis.MEAN) # kg
oc_cube_global_annual_mean = oc_cube_area_sum.aggregated_by('year', iris.analysis.MEAN) # kg

# %%
so2_cube_global_annual_mean.data  # kg

# %%
so4_cube_global_annual_mean.data  # kg

# %%
bc_cube_global_annual_mean.data  # kg

# %%
oc_cube_global_annual_mean.data  # kg

# %%
so2_cube_area_sum.data

# %%
