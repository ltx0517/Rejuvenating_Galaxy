#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 10:46:54 2025

@author: txl
"""
import numpy as np
import proplot as pplt
import os
import glob
from datetime import date
today_str = date.today().strftime("%m%d")

def binning(x,y):
    x, y = np.array(x), np.array(y)
    bins = np.linspace(0, 2, 11)

    digitized = np.digitize(x, bins)
    binned_y  = [y[digitized == i] for i in range(1, len(bins))]

    medians       = [np.median(bin_y) for bin_y in binned_y]
    percentile_16 = [np.percentile(bin_y, 16) for bin_y in binned_y]
    percentile_84 = [np.percentile(bin_y, 84) for bin_y in binned_y]

    return bins, medians, percentile_16, percentile_84

# Get data in folder and load txt data by glob and numpy
folder_sf    = '/Volumes/KINGSTON/MaNGA/1D_Data/Profile/Z'
all_files_sf = glob.glob(os.path.join(folder_sf, "*.txt"))
print(len(all_files_sf))
R_sf, Z_sf = [], [] # Container for restore all the data point. This will be used to calculate the mean profile.
fig, ax = pplt.subplots(figsize=(6,4), facecolor='none', dpi=250)
# Plotting style setting
pplt.rc['font.family'] = 'serif'
pplt.rc['axes.labelsize'] = 12
pplt.rc['savefig.facecolor'] = 'none'
pplt.rc['savefig.transparent'] = True
pplt.rc['savefig.format'] = 'pdf'
pplt.rc.save()

bad_profile_gal = 0
for i, file in enumerate(all_files_sf):
    data = np.loadtxt(file) 
    if len(data) > 3:
        rr, zz = data[:, 0], data[:, 1]
        ax.plot(rr, zz, lw=0.3, alpha=0.5, color='light gray', zorder=0)
        R_sf.extend(rr)
        Z_sf.extend(zz)
    else:
        bad_profile_gal += 1
        print(f"=== File {file} has too few data points, skipping. ===")
    #if i == 20: break

bins, med, b, u = binning(R_sf, Z_sf)
for i in range(len(bins)-1):
    if i == 1: 
        ax.fill_between([bins[i], bins[i+1]], b[i], u[i], color='blue2', alpha=0.5, label='16% & 84% of Control SF', zorder=1)
    else:
        ax.fill_between([bins[i], bins[i+1]], b[i], u[i], color='blue2', alpha=0.5, zorder=1)

file_rej = '/Volumes/KINGSTON/MaNGA/1D_Data/Profile/Rej_scatter/All_scatter_rej.txt'
file_sf  = '/Volumes/KINGSTON/MaNGA/1D_Data/Profile/Rej_scatter/All_scatter_sf.txt'
R_rej, Z_rej = np.loadtxt(file_rej, unpack=True)
R_rej_sf, Z_rej_sf = np.loadtxt(file_sf, unpack=True)

bins_rej, med_rej, b_rej, u_rej = binning(R_rej, Z_rej)
bins_rej_sf, med_rej_sf, b_rej_sf, u_rej_sf = binning(R_rej_sf, Z_rej_sf)

#fig, ax = pplt.subplots()
#ax.hist(Z_rej_sf)
#pplt.show()

# The marginal distribution
px = ax.panel('t', space=0)
py = ax.panel('r', space=0)
px.hist(R_rej, bins=pplt.arange(0, 2, 0.05), color='green4', alpha=0.8)
py.hist(Z_rej, bins=pplt.arange(8.35, 8.65, 0.01), color='green4', alpha=0.8, orientation='horizontal')

ax.plot(bins[:-1]+0.1, med, lw=1.5, color='bright blue', label='Control SFG SF Median')
ax.scatter(R_rej, Z_rej, alpha=0.5, facecolor='none', color= 'green4', s=4, label='RJ Spaxels', zorder=1)
ax.plot(bins_rej[:-1]+0.1, med_rej, lw=1.5, color='green8', label='RJGs RG Median')
ax.plot(bins_rej_sf[:-1]+0.1, med_rej_sf, lw=1.5, color='orange', label='RJGs SF Spaxels')
ax.legend(ncol=2, fontsize='small')
ax.format(xlabel='Effective radius [R$_e]$', ylabel='12 + log(O/H)', xlabelsize=18, ylabelsize=18, ylim=(8.35, 8.65), suptitle='Gas-Phase Metallicity Profiles')
fig.savefig(f'/Users/txl/Desktop/MaNGA/Paper_figures/met_profiles_sf_mask_v4_{today_str}.pdf', bbox_inches='tight', transparent=True, facecolor='none')
pplt.show()