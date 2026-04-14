#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
# Read the CSV file
file='../output/ghg_concentrations.csv'
df=pd.read_csv(file)


# %%
# Filter the data for CO2, CH4, and N2O concentrations and years 2000 to 2025
data = df[(df['YYYY'] >= 2000) & (df['YYYY'] <= 2025)][['YYYY', 'CO2', 'CH4', 'N2O']]

data2 = df[(df['YYYY'] >= 2000) & (df['YYYY'] <= 2025)][['YYYY', 'CFC-12', 'CFC-11', 'CFC-113', 'CH3CCl3', 'CCl4']]
hfcs = df[(df['YYYY'] >= 2000) & (df['YYYY'] <= 2025)][['YYYY', 'HFC-134a', 'HFC-23', 'HFC-32', 'HFC-125', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-236fa', 'HFC-245fa', 'HFC-365mfc', 'HFC-43-10mee']]
pfcs = df[(df['YYYY'] >= 2000) & (df['YYYY'] <= 2025)][['YYYY', 'CF4','C2F6','C3F8','c-C4F8', 'n-C4F10', 'n-C5F12','n-C6F14','i-C6F14','C7F16','C8F18']]
pfcs['PFCs']=pfcs['CF4'] + pfcs['C2F6'] +pfcs['C3F8'] +pfcs['c-C4F8']
hfcs['otherHFCs']=hfcs['HFC-23']+hfcs['HFC-32'] +hfcs['HFC-125'] +hfcs['HFC-143a']+ hfcs['HFC-152a']+ hfcs['HFC-227ea'] +hfcs['HFC-236fa'] +hfcs['HFC-245fa'] +hfcs['HFC-365mfc']+hfcs['HFC-43-10mee']
data3= df[(df['YYYY'] >= 2000) & (df['YYYY'] <= 2025)][['YYYY', 'HCFC-22', 'CH2Cl2']]

# %%
# Use a color-blind-friendly palette
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

colors = {
    'CO2': '#E69F00',     # orange
    'CH4': '#56B4E9',     # blue
    'N2O': '#009E73',     # green
    'CFC-12': '#F0E442',  # yellow
    'CFC-11': '#0072B2',  # dark blue
    'CCl4': '#D55E00',    # vermillion
    'CFC-113': '#CC79A7', # pink
    'CH3CCl3': '#999999', # grey
    'HCFC-22': '#E69F00',
    'HFC-134a': '#56B4E9',
    'other HFCs': '#009E73',
    'PFCs': '#F0E442',
    'CH2Cl2': '#0072B2',
}

fig, (ax1, ax4, ax5) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, gridspec_kw={'hspace': 0.05})


# --- Top plot ---
line1, = ax1.plot(data['YYYY'], data['CO2'], label='CO$_{2}$ (ppm)', color=colors['CO2'], lw=2.5)
ax1.set_ylabel('CO$_{2}$ (ppm)', fontsize=12)
ax1.tick_params(axis='y', labelcolor=colors['CO2'], labelsize=10)
ax1.set_title('Greenhouse Gas Concentrations (2000–2025)', fontsize=14)
# Add vertical grid lines only on ax1
ax1.xaxis.grid(True, which='major', linestyle='--', alpha=0.5)
ax1.yaxis.grid(False)  # Turn off horizontal grid lines

ax1.set_xlim([2000, 2025])
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax1.set_ylim(350, 450)
plt.xticks(rotation=45)

ax2 = ax1.twinx()
line2, = ax2.plot(data['YYYY'], data['CH4'], label='CH$_{4}$ (ppb)', color=colors['CH4'], lw=2.5, linestyle='--', marker='o')
ax2.set_ylabel('CH$_{4}$ (ppb)', fontsize=12)
ax2.tick_params(axis='y', labelcolor=colors['CH4'], labelsize=10)
ax2.set_ylim(1700, 2000)

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
line3, = ax3.plot(data['YYYY'], data['N2O'], label='N$_{2}$O (ppb)', color=colors['N2O'], lw=2.5, linestyle='-.', marker='s')
ax3.set_ylabel('N$_{2}$O (ppb)', fontsize=12)
ax3.tick_params(axis='y', labelcolor=colors['N2O'], labelsize=10)
ax3.set_ylim(310, 350)

ax1.axvspan(2020, 2025, color='gray', alpha=0.2)

# Collect all lines for a combined legend
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left", fontsize=10)

# --- Middle plot ---
ax4.plot(data['YYYY'], data2['CFC-12'], label='CFC-12', color=colors['CFC-12'], lw=2)
ax4.plot(data['YYYY'], data2['CFC-11'], label='CFC-11', color=colors['CFC-11'], lw=2)
ax4.plot(data['YYYY'], data2['CCl4'], label='CCl$_{4}$', color=colors['CCl4'], lw=2)
ax4.plot(data['YYYY'], data2['CFC-113'], label='CFC-113', color=colors['CFC-113'], lw=2)
ax4.plot(data['YYYY'], data2['CH3CCl3'], label='CH$_{3}$CCl$_{3}$', color=colors['CH3CCl3'], lw=2)

ax4.set_ylabel('Concentration (ppt)', fontsize=12)
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend(fontsize=10,loc="upper left")
ax4.axvspan(2020, 2025, color='gray', alpha=0.2)

# --- Bottom plot ---
ax5.plot(data['YYYY'], data3['HCFC-22'], label='HCFC-22', color=colors['HCFC-22'], lw=2)
ax5.plot(data['YYYY'], hfcs['HFC-134a'], label='HFC-134a', color=colors['HFC-134a'], lw=2, linestyle='--')
ax5.plot(data['YYYY'], hfcs['otherHFCs'], label='Other HFCs', color=colors['other HFCs'], lw=2)
ax5.plot(data['YYYY'], pfcs['PFCs'], label='PFCs', color=colors['PFCs'], lw=2)
ax5.plot(data['YYYY'], data3['CH2Cl2'], label='CH$_{2}$Cl$_{2}$', color=colors['CH2Cl2'], lw=2, linestyle=':')

ax5.set_ylabel('Concentration (ppt)', fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(True, linestyle='--', alpha=0.5)
ax5.axvspan(2020, 2025, color='gray', alpha=0.2)

plt.xlabel('Year', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)  # Optional: fine-tune again after tight_layout

# Save outputs
plt.savefig("../plots/GHG_concentrations_accessible.png", dpi=300)
plt.savefig("../plots/GHG_concentrations_accessible.pdf")
plt.show()


# %%
#This next part makes table S2 in the supplement
df = df.set_index('YYYY')


# %%
#code taken from https://github.com/ClimateIndicator/forcing-timeseries/blob/main/notebooks/01_trace-gas-global-mean.py
# Piers Forster 23 April 2005

# %% [markdown]
# ## Aggregated categories

# %%
gases_hfcs = [
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
]
gases_montreal = [
    'CFC-12',
    'CFC-11',
    'CFC-113',
    'CFC-114',
    'CFC-115',
    'CFC-13',
    'CFC-112',
    'CFC-112a',
    'CFC-113a',
    'CFC-114a',
    'HCFC-22',
    'HCFC-141b',
    'HCFC-142b',
    'HCFC-133a',
    'HCFC-31',
    'HCFC-124',
    'CH3CCl3',
    'CCl4',  # yes
    'CH3Cl',  # no
    'CH3Br',  # yes
    'CH2Cl2',  # no!
    'CHCl3',  # no
    'Halon-1211',
    'Halon-1301',
    'Halon-2402',

]
gases_pfc = [
    'CF4',
    'C2F6',
    'C3F8',
    'c-C4F8',
    'n-C4F10',
    'n-C5F12',
    'n-C6F14',
    'i-C6F14',
    'C7F16',
    'C8F18',
]

# %%
# source: Hodnebrog et al 2020 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019RG000691
radeff = {
    'HFC-125':      0.23378,
    'HFC-134a':     0.16714,
    'HFC-143a':     0.168,
    'HFC-152a':     0.10174,
    'HFC-227ea':    0.27325,
    'HFC-23':       0.19111,
    'HFC-236fa':    0.25069,
    'HFC-245fa':    0.24498,
    'HFC-32':       0.11144,
    'HFC-365mfc':   0.22813,
    'HFC-43-10mee': 0.35731,
    'NF3':          0.20448,
    'C2F6':         0.26105,
    'C3F8':         0.26999,
    'n-C4F10':      0.36874,
    'n-C5F12':      0.4076,
    'n-C6F14':      0.44888,
    'i-C6F14':      0.44888,
    'C7F16':        0.50312,
    'C8F18':        0.55787,
    'CF4':          0.09859,
    'c-C4F8':       0.31392,
    'SF6':          0.56657,
    'SO2F2':        0.21074,
    'CCl4':         0.16616,
    'CFC-11':       0.25941,
    'CFC-112':      0.28192,
    'CFC-112a':     0.24564,
    'CFC-113':      0.30142,
    'CFC-113a':     0.24094, 
    'CFC-114':      0.31433,
    'CFC-114a':     0.29747,
    'CFC-115':      0.24625,
    'CFC-12':       0.31998,
    'CFC-13':       0.27752,
    'CH2Cl2':       0.02882,
    'CH3Br':        0.00432,
    'CH3CCl3':      0.06454,
    'CH3Cl':        0.00466,
    'CHCl3':        0.07357,
    'HCFC-124':     0.20721,
    'HCFC-133a':    0.14995,
    'HCFC-141b':    0.16065,
    'HCFC-142b':    0.19329,
    'HCFC-22':      0.21385,
    'HCFC-31':      0.068,
    'Halon-1202':   0,       # not in dataset
    'Halon-1211':   0.30014,
    'Halon-1301':   0.29943,
    'Halon-2402':   0.31169,
    'CO2':          0,       # different relationship
    'CH4':          0,       # different relationship
    'N2O':          0        # different relationship
}


# %%
pfc_hfc134a_eq_1750 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_1750 = pfc_hfc134a_eq_1750 + (df.loc[1750, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_1750 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_1750 = hfc_hfc134a_eq_1750 + (df.loc[1750, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_1750 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_1750 = montreal_cfc12_eq_1750 + (df.loc[1750, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_1750, hfc_hfc134a_eq_1750, montreal_cfc12_eq_1750


# %%
pfc_hfc134a_eq_1850 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_1850 = pfc_hfc134a_eq_1850 + (df.loc[1850, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_1850 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_1850 = hfc_hfc134a_eq_1850 + (df.loc[1850, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_1850 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_1850 = montreal_cfc12_eq_1850 + (df.loc[1850, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_1850, hfc_hfc134a_eq_1850, montreal_cfc12_eq_1850

# %%
#  note that PFCs are in CF4 units%% 
pfc_hfc134a_eq_2019 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2019 = pfc_hfc134a_eq_2019 + (df.loc[2019, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2019 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2019 = hfc_hfc134a_eq_2019 + (df.loc[2019, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2019 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2019 = montreal_cfc12_eq_2019 + (df.loc[2019, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2019, hfc_hfc134a_eq_2019, montreal_cfc12_eq_2019

# %%
pfc_hfc134a_eq_2022 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2022 = pfc_hfc134a_eq_2022 + (df.loc[2022, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2022 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2022 = hfc_hfc134a_eq_2022 + (df.loc[2022, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2022 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2022 = montreal_cfc12_eq_2022 + (df.loc[2022, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2022, hfc_hfc134a_eq_2022, montreal_cfc12_eq_2022

# %%
pfc_hfc134a_eq_2023 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2023 = pfc_hfc134a_eq_2023 + (df.loc[2023, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2023 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2023 = hfc_hfc134a_eq_2023 + (df.loc[2023, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2023 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2023 = montreal_cfc12_eq_2023 + (df.loc[2023, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2023, hfc_hfc134a_eq_2023, montreal_cfc12_eq_2023

# %%
pfc_hfc134a_eq_2024 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2024 = pfc_hfc134a_eq_2024 + (df.loc[2024, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2024 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2024 = hfc_hfc134a_eq_2024 + (df.loc[2024, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2024 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2024 = montreal_cfc12_eq_2024 + (df.loc[2024, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2024, hfc_hfc134a_eq_2024, montreal_cfc12_eq_2024

# %%
pfc_hfc134a_eq_2025 = 0
for gas in gases_pfc:
    pfc_hfc134a_eq_2025 = pfc_hfc134a_eq_2025 + (df.loc[2025, gas] * radeff[gas] / radeff['CF4'])
hfc_hfc134a_eq_2025 = 0
for gas in gases_hfcs:
    hfc_hfc134a_eq_2025 = hfc_hfc134a_eq_2025 + (df.loc[2025, gas] * radeff[gas] / radeff['HFC-134a'])
montreal_cfc12_eq_2025 = 0
for gas in gases_montreal:
    montreal_cfc12_eq_2025 = montreal_cfc12_eq_2025 + (df.loc[2025, gas] * radeff[gas] / radeff['CFC-12'])

# %%
pfc_hfc134a_eq_2025, hfc_hfc134a_eq_2025, montreal_cfc12_eq_2025

# %%
df_t=df.transpose()
df_select=df_t[[1750,1850,2019,2023,2024,2025]]


# %%
neworder=['CO2','CH4','N2O','NF3','SF6','SO2F2']+['HFCs_total']+gases_hfcs+['PFCs_total']+ gases_pfc+ ['Montreal_total']+gases_montreal


# %%
df_s_order=df_select.reindex(neworder)


# %%
#insert sums, note PFCs are in CF4 units not HFC134a
df_s_order.loc['Montreal_total',[1750,1850,2019,2023,2024,2025]] = \
    [montreal_cfc12_eq_1750, montreal_cfc12_eq_1850, \
     montreal_cfc12_eq_2019, montreal_cfc12_eq_2023,\
     montreal_cfc12_eq_2024,montreal_cfc12_eq_2025]

df_s_order.loc['HFCs_total',[1750,1850,2019,2023,2024,2025]] = \
    [hfc_hfc134a_eq_1750, hfc_hfc134a_eq_1850, \
     hfc_hfc134a_eq_2019, hfc_hfc134a_eq_2023,\
     hfc_hfc134a_eq_2024,hfc_hfc134a_eq_2025]

df_s_order.loc['PFCs_total',[1750,1850,2019,2023,2024,2025]] = \
    [pfc_hfc134a_eq_1750, pfc_hfc134a_eq_1850, \
     pfc_hfc134a_eq_2019, pfc_hfc134a_eq_2023,\
     pfc_hfc134a_eq_2024,pfc_hfc134a_eq_2025]


# %%
#print to one decimal place for table S2
print(df_s_order.round(1))


# %%
df_s_order


# %%
