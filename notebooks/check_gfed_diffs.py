# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Download and process biomass burning emissions
#
# New for 2025: The GFED data is now on an FTP server and can't be directly downloaded over HTTP. Therefore, use an SFTP client and log in thus:
#
# https://www.globalfiredata.org/ancill/GFED5_SFTP_info.txt
#
# GFED4.1s is still available at this location and we will use it.
#
# However, we note that some of the files (2015, 2017, 2018, 2019) are different to the versions previously released at https://www.geo.vu.nl/~gwerf/GFED/GFED4/. 
#
# Furthermore, 2017 is corrupted.
#
# For now, we will use the HTTPS versions up to 2023, and SFTP versions for 2024 and 2025.

# %%
import os
import hashlib

import h5py
import pooch

# %%
files_gwerf = {}
files_sftp = {}

for year in range(1997, 2017):
    files_gwerf[year] = f"../data/slcf_emissions/gfed/gfed_4.1s_gwerf/GFED4.1s_{year}.hdf5"
    files_sftp[year] = f"../data/slcf_emissions/gfed/gfed_4.1s_sftp/GFED4.1s_{year}.hdf5"
    
for year in range(2017, 2024):
    files_gwerf[year] = f"../data/slcf_emissions/gfed/gfed_4.1s_gwerf/GFED4.1s_{year}_beta.hdf5"
    files_sftp[year] = f"../data/slcf_emissions/gfed/gfed_4.1s_sftp/GFED4.1s_{year}_beta.hdf5"

for year in range(2024, 2026):
    files_sftp[year] = f"../data/slcf_emissions/gfed/gfed_4.1s_sftp/GFED4.1s_{year}_beta.hdf5"

# %%
for year in range(1997, 2024):
    with open(files_gwerf[year], "rb") as f:
        hash_gwerf = hashlib.file_digest(f, "md5").hexdigest()
    with open(files_sftp[year], "rb") as f:
        hash_sftp = hashlib.file_digest(f, "md5").hexdigest()
    print(year, hash_gwerf == hash_sftp)

# %%
hashes = {
    'emissions_factors': '5f68c5c4ffdb7d81d3d2fefa662dcad9dd66f2b4097350a08a045523626383b2',
}

files = {}

for year in range(1997, 2017):
    files[year] = f"../data/slcf_emissions/gfed/gfed_4.1s_sftp/GFED4.1s_{year}.hdf5"

for year in range(2017, 2026):
    files[year] = f"../data/slcf_emissions/gfed/gfed_4.1s_sftp/GFED4.1s_{year}_beta.hdf5"

files['emissions_factors'] = pooch.retrieve(
    "https://www.geo.vu.nl/~gwerf/GFED/GFED4/ancill/GFED4_Emission_Factors.txt",
    hashes['emissions_factors']
)

# %%
with h5py.File(files[2017], 'r') as f:
    f['emissions/12/DM'][:]

# %%
