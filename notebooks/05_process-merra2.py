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
# # Process MERRA2 reanalysis for aerosol emissions
#
# The purpose here is to attempt to extend CEDS beyond 2022 using MERRA reanalysis.
#
# The dataset we are using is MERRA2_MONTHLY/M2TMNXADG.5.12.4. They can be obtained from NASA EarthData.
#
# These are downloaded to data/slcf_emissions/merra. They are not added to the repository owing to file size.
#
# Variables:
# - OCEMAN
# - BCEMAN
# - SO4EMAN
#
# We can also look at (to extend BB4CMIP to 2024):
# - OCEMBB
# - BCEMBB
# - SO4EMBB
#
# **Now for some weird reason there is no difference in emissions between years! This is not a coding error in the below, I've checked it out using ncdiff**

# %%
import iris
import iris.coord_categorisation as cat
from iris.util import equalise_attributes, unify_time_units
import numpy as np

# %%
# this takes a long time and they are not huge files. I should go and break them up manually into single variables with ncks
so2_cubes = iris.load(
    '../data/slcf_emissions/merra/MERRA2_400.tavgM_2d_adg_Nx.*.nc4',
    constraints = "SO2 Anthropogenic Emissions __ENSEMBLE__"
)

# %%
so4_cubes = iris.load(
    '../data/slcf_emissions/merra/MERRA2_400.tavgM_2d_adg_Nx.*.nc4',
    constraints = "SO4 Anthropogenic Emissions __ENSEMBLE__"
)

# %%
unify_time_units(so2_cubes)
unify_time_units(so4_cubes)

# %%
equalise_attributes(so2_cubes)
equalise_attributes(so4_cubes)

# %%
# Remove all the other attributes.
for cube in so2_cubes:
    coord = cube.coord('time')
    del coord.attributes["begin_date"]
    del coord.attributes["begin_time"]
    del coord.attributes["time_increment"]
    del coord.attributes["valid_range"]
    del coord.attributes["vmax"]
    del coord.attributes["vmin"]

for cube in so4_cubes:
    coord = cube.coord('time')
    del coord.attributes["begin_date"]
    del coord.attributes["begin_time"]
    del coord.attributes["time_increment"]
    del coord.attributes["valid_range"]
    del coord.attributes["vmax"]
    del coord.attributes["vmin"]

# %%
so2_cube = so2_cubes.concatenate_cube()
so4_cube = so4_cubes.concatenate_cube()

# %%
# no bounds included so guess them
so2_cube.coord('latitude').guess_bounds()
so2_cube.coord('longitude').guess_bounds()
so4_cube.coord('latitude').guess_bounds()
so4_cube.coord('longitude').guess_bounds()

# %%
so2_area_weights = iris.analysis.cartography.area_weights(so2_cube)
so4_area_weights = iris.analysis.cartography.area_weights(so4_cube)

# %%
cat.add_year(so2_cube, 'time', name='year')
cat.add_year(so4_cube, 'time', name='year')
so2_cube_area_sum = so2_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=so2_area_weights)
so4_cube_area_sum = so4_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=so4_area_weights)
so2_cube_global_annual_mean = so2_cube_area_sum.aggregated_by('year', iris.analysis.MEAN)  # kg s-1 units
so4_cube_global_annual_mean = so4_cube_area_sum.aggregated_by('year', iris.analysis.MEAN)  # kg s-1 units

# %%
so2_cube_global_annual_mean.data * 365 * 24 * 60 * 60  # kg yr-1

# %%
so4_cube_global_annual_mean.data * 365 * 24 * 60 * 60  # kg yr-1

# %%
so2_cube[:, 180, 288].data

# %%
cube

# %%
