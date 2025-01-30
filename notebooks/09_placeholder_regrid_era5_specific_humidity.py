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
# # Regrid ERA5 specific humidity
#
# The intention is to use a dataset that can give us an estimate of the water vapour loading from HTHH.
#
# At the time of writing on 29.1.2025, the MLS data does not go beyond the end of 2023, and may not be in existance much longer.
#
# ERA5 does not appear to assimilate HTHH, and this paper also suggests my understanding is correct: https://egusphere.copernicus.org/preprints/2022/egusphere-2022-517/egusphere-2022-517-manuscript-version6.pdf
#
# As a workaround, we can use the 2023 estimate and extrapolate it forward.

# %%
