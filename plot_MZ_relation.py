#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Sat Mar 07 12:10:17 2024

@author: txl
"""
from MyTool import Maps, Pipe3D, Drpall, reju_gal
from MyTool_2 import binning
import numpy as np
from datetime import date
import matplotlib.patches as patches
today_str = date.today().strftime("%m%d")

# Rejuvenations
reju_mas, reju_lig = reju_gal()[1:]
reju_int = ['11948-3704', '11963-9101', '12080-12705', '12624-3701', '8618-12703', '8724-12704']

""" For integrated rejuvenating galaxies """
M_int, Z_int, S_int = [], [], []
for gal in reju_int:
    maps, pips, drp = Maps(gal), Pipe3D(gal), Drpall(gal)
    # Calculate the metallicity and mass density
    Z = maps.metallicity()[0]
    #SFR = maps.SFR()[0]
    M, M_mask = pips.sigma()
    M = np.log10(10**M * drp.ba())
    
    # Get the relative masks
    Ha_mask  = maps.HaEW()[1]     # The recently star-forming mask
    Z_mask   = np.ma.getmask(Z) # The metallicity mask
    BPT_mask = maps.BPT_sf_mask()[2] # The BPT sf mask
    rej_mask = maps.rejuvenated(adjacent_spaxel=5)[2]
    
    mask = Ha_mask | Z_mask | BPT_mask | M_mask #| rej_mask
    
    #print("===== Global RJG %s metallicity: (%.2f, %.2f) ====="%(gal, np.min(np.ma.array(Z, mask=mask)), np.max(np.ma.array(Z, mask=mask))))
    
    #if np.min(np.ma.array(Z, mask=mask)) < 8.3:
    #    print("===== Global RJG %s have low metallicity. ====="%gal)
    
    # Mask M and Z
    M_int.append(np.ma.array(M, mask = mask).compressed())
    Z_int.append(np.ma.array(Z, mask = mask).compressed())
    #S_int.append(np.ma.array(SFR, mask = mask).compressed())

""" For spatially-resoved rejuvenating galaxies """
problem_gal   = []
zero_data_gal = []
M_spa, Z_spa, S_spa = [], [], []
for gal in reju_mas:
    try:
        maps, pips, drp = Maps(gal), Pipe3D(gal), Drpall(gal)
        # Calculate the star-formation rate (Ha as a proxy)
        # Ha = maps.get_masked('Ha-6564')[1]
        # Hb = maps.get_masked('Hb-4862')[1]
        # La = maps.luminosity(Ha)
        # spaxel_area_pc = maps.spaxel_area_pc()
        # La = np.log10(La/spaxel_area_pc) # Unit: ??/pc
        
        # SFR = maps.SFR(Ha, Hb)
        
        # Calculate the metallicity and mass density
        Z = maps.metallicity()[0]
        #print(gal)
        M, M_mask = pips.sigma()
        M = np.log10(10**M * drp.ba())
    
        # Get the relative masks
        Ha_mask  = maps.HaEW()[1]     # The recently star-forming mask
        Z_mask   = np.ma.getmask(Z) # The metallicity mask
        BPT_mask = maps.BPT_sf_mask()[2] # The BPT sf mask
        rej_mask = maps.rejuvenated()[2] # The rejuvenated mask
        
        mask = Ha_mask | Z_mask | BPT_mask | M_mask #| rej_mask # Modify here if needed
        
        # Compress to 1D
        M_com = np.ma.array(M, mask = mask).compressed()
        Z_com = np.ma.array(Z, mask = mask).compressed()
        
        if len(M_com) > 0:
            MZ_combine = np.column_stack((M_com, Z_com))
            np.savetxt('/Volumes/KINGSTON/MaNGA/1D_Data/MZ_relation/SF_Spaxel/%s.txt'%gal, MZ_combine, fmt='%.3f', delimiter='\t')
            print("===== Global RJG %s metallicity: (%.2f, %.2f) =====" %(gal, np.min(Z_com), np.max(Z_com)))
            # Mask M and Z
            M_spa.append(M_com)
            Z_spa.append(Z_com)
            #S_spa.append(np.ma.array(SFR, mask = mask).compressed())
        else:
            zero_data_gal.append(gal)
    except:
        problem_gal.append(gal)
    
"""" Do our control sample: 508 face-on & blue spiral galaxies """



"""" Binning part """
M_spa, M_int = np.concatenate(M_spa), np.concatenate(M_int) # Add int here if needed
Z_spa, Z_int = np.concatenate(Z_spa), np.concatenate(Z_int) # Add int here if needed
#M_spa, Z_spa = np.concatenate(M_spa), np.concatenate(Z_spa) # Add int here if needed
# S_spa = np.concatenate(S_spa) # Add int here if needed

# Binning and Bootstrap for the medians
B_int, med_int, err_int = binning(M_int, Z_int, num = 10)
B_spa, med_spa, err_spa = binning(M_spa, Z_spa, num = 13) # Metallicity part
#B2_spa, med2_spa, err2_spa = binning(M_spa, S_spa, num = 9) # SFR part

# Plot part
import proplot as pplt
import matplotlib.pyplot as plt
pplt.rc.update({'font.family': 'TeX Gyre Schola', 'savefig.bbox': 'tight'})
import matplotlib.patches as mpatches
#import seaborn as sns


a, b, c = 8.55, 0.014, 3.14      # B&B best fitting (Metallicity)
A, B, C = 8.64, -1.1917, -1.589  # Our massive best fit
X  = np.linspace(0.1, 3.75, 100)
Y0 = a + b*(X-c)*np.exp(-(X-c))
Y1 = A + B*(X-C)*np.exp(-(X-C))

# SFMS spatially-resolved fitting
#X = np.linspace(2.0, 3.2, 100)
#Y = 0.72*X

# Plot part
fig, ax = pplt.subplots(figsize=(4.5,4))
#ax.hist2d(M_spa, Z_spa, bins=30, cmap='Blues', label='Star-forming spaxels')
#sns.kdeplot(x=M_int, y=Z_int, cmap='Blues', label = 'Integrated')
#sns.kdeplot(x=M_spa, y=Z_spa, cmap='Reds', label = 'Spatially-resolved')
ax.scatter(M_spa, Z_spa, ms=3, marker='o', alpha = 0.3, color = 'pastel blue')
ax.scatter(M_int, Z_int, ms=3, marker='o', alpha = 0.4, color = 'yellow4')
ax.errorbar(B_spa, med_spa, yerr=err_spa, fmt='o', capsize=5, ms=5, label='Local selection (%i)'%len(M_spa), color='blue8')
ax.errorbar(B_int, med_int, yerr=err_int, fmt='o',color='yellow8', capsize=5, ms=5, label='Global selection (%i)' %len(M_int))
ax.plot(X, Y0, lw=1, color='k', label='B&B et al. 2016')
ax.plot(X, Y1, lw=1, color='gray', ls='-.', label='Massive fitting')

# The marginal histograms
px = ax.panel('t', space=0)
py = ax.panel('r', space=0)

px.hist(M_spa, bins=pplt.arange(-0.1, 3.7, 0.2), color='pastel blue', alpha=0.7, density=True)
px.hist(M_int, bins=pplt.arange(-0.1, 3.7, 0.2), color='yellow4', alpha=0.5, density=True)

py.hist(Z_spa, bins=pplt.arange(8.1, 8.7, 0.01), color='pastel blue', alpha=0.7, orientation='horizontal', density=True)
py.hist(Z_int, bins=pplt.arange(8.1, 8.7, 0.02), color='yellow4', alpha=0.5, orientation='horizontal', density=True)

px.yaxis.set_label_position('right')
px.yaxis.tick_right()
py.xaxis.set_label_position('top')
py.xaxis.tick_top()

# Turn off the tick labels for the histograms
px.yaxis.set_ticklabels([])
py.xaxis.set_ticklabels([])

#xrange1, yrange1 = [1.75, 2.5, 2.5, 1.75, 1.75], [8.21, 8.21, 8.33, 8.33, 8.21]
#ax.fill_between(xrange1, yrange1, alpha=0.1, color='bright lime green', zorder=2, label='12080-12705')
#xrange2, yrange2 = [2.85, 3.3, 3.3, 2.85], [8.37, 8.37, 8.45, 8.45]
#ax.fill_between(xrange2, yrange2, alpha=0.1, color='vivid green', zorder=2, label='8452-1902')
#rect1 = patches.Rectangle((1.75, 8.21), 0.75, 0.12, alpha=0.1, facecolor='bright lime green')
#rect2 = patches.Rectangle((2.85, 8.37), 0.45, 0.08, alpha=0.1, facecolor='light purple')
#ax.add_patch(rect1)
#ax.add_patch(rect2)
#ax.text(1.76, 8.34, '12080-12705', va='center', ha='left', fontsize=7, color='bright lime green')
#ax.text(2.86, 8.46, '8452-1902',   va='center', ha='left', fontsize=7, color='light purple')

#handles = [mpatches.Patch(facecolor=plt.cm.Blues(100), label = 'Integrated (%i)' %len(M_int)),
#           mpatches.Patch(facecolor=plt.cm.Reds(100), label = 'Spatially-resolved (%i)' %len(M_spa))]
#plt.legend(handles=handles, loc='upper left')

ax.legend(ncols=1, fontsize=10, loc='ll')
#ax.set_xlabel('Stellar Mass Density [log (M$_\odot$/pc$^2$)]', fontsize=14)
#ax.set_ylabel('12 + log (O/H)', fontsize=14)
ax.set_xlim(0.1, 3.4)
ax.set_ylim(8.2, 8.7)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.format(xlabel='Stellar Mass Density [log (M$_\odot$/pc$^2$)]', ylabel='12 + log (O/H)', ylim=(8.1, 8.7), xlim=(-0.1, 3.7),
          xlabelsize=14, ylabelsize=14, title='Star-Forming Spaxels')
#fig.format(suptitle='Rejuvenating Spaxels', fontsize=12)
#plt.ylabel('L$_{Ha}$ [log(erg/s/pc$^2$)]') 
#plt.xlim(0.0, 3.5)
fig.tight_layout()
fig.savefig(f'/Users/txl/Desktop/MaNGA/Paper_figures/MZ_relations_sf_{today_str}.pdf', transparent=True, facecolor='none', tight_layout=True)
pplt.show()

#print('%i galaxies do not have valid data.'%len(problem_gal))
#print('%i galaxies have no qualified emission line estimate.'%len(zero_data_gal))