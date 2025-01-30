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
# # This script splits up the MERRA2 files into individual variables
#
# The purpose here is to attempt to extend CEDS beyond 2022 using MERRA reanalysis.
#
# The dataset we are using is MERRA2_MONTHLY/M2TMNXAER.5.12.4. They can be obtained from NASA EarthData.
#
# These are downloaded to data/slcf_emissions/merra. They are not added to the repository owing to file size.
#
# We use column burdens rather than emissions, because emissions do not appear to vary year-to-year.
#
# Also, it is vital to rename files from 401 to 400

# %% language="bash"
# ncrcat -O -v BCCMASS ../data/slcf_emissions/merra/MERRA2_400.tavgM_2d_aer_Nx.*.nc4 ../data/slcf_emissions/merra/BCCMASS.nc
# ncrcat -O -v OCCMASS ../data/slcf_emissions/merra/MERRA2_400.tavgM_2d_aer_Nx.*.nc4 ../data/slcf_emissions/merra/OCCMASS.nc
# ncrcat -O -v SO2CMASS ../data/slcf_emissions/merra/MERRA2_400.tavgM_2d_aer_Nx.*.nc4 ../data/slcf_emissions/merra/SO2CMASS.nc
# ncrcat -O -v SO4CMASS ../data/slcf_emissions/merra/MERRA2_400.tavgM_2d_aer_Nx.*.nc4 ../data/slcf_emissions/merra/SO4CMASS.nc

# %%
