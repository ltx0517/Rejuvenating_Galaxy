#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 2023
Lateset on Tue Dec 11 2023

@author: txl
"""
"""
This script is to plot all the relative properties of a rejuvenated galaxies.
(1) The optical image
(2) The rejuvenated region inside galaxies (for int. selection, SF region)
(3) The Σ-Z relation of the rejuvenated/star-forming spaxels
(4) The metallicity gradient and profile (with star-forming rate)
(5) The SFR profile
(6) BPT diagram (NII only, other two lines are too weak)
"""
#from MyTool import get_optical_image, rejuvenated, BPT_sf_mask, sigma, metallicity, HaEW, MG, reju_gal, SFR
from MyTool import Maps, Pipe3D
from MyTool_2 import MG, reju_gal
from scipy.stats import spearmanr, linregress
import matplotlib.pyplot as plt
import numpy as np

rej_spa, rej_int = reju_gal()

# B&B et al. 2016
a, b, c = 8.55, 0.014, 3.14
X = np.linspace(1.3, 3.25, 100)
Y = a + b*(X-c)*np.exp(-(X-c))

for gal in rej_int[9:]:
    maps, pips = Maps(gal), Pipe3D(gal)
    fig, ax = plt.subplots(2, 3, figsize = (18, 10), dpi=250)
    fig.suptitle('%s'%gal, fontsize = 26)
    
    # (1) Optical image
    image = maps.get_optical_image()
    
    ax[0,0].imshow(image.data)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    
    # (2) Rejuvenated region (spa gal) / Star-forming region (int gal) (line 46)
    rej_mask = maps.rejuvenated()[2] # The rejuvenated mask
    x, y, bpt_mask = maps.BPT_sf_mask()    # The BPT diagram
    Ha, Ha_mask = maps.HaEW()        # The recently star-forming mask
    
    im = ax[0,1].imshow(np.ma.array(Ha, mask = (bpt_mask | Ha_mask)), origin = 'lower')
    cbar = fig.colorbar(im, ax=ax[0,1])
    ax[0,1].set_title('Star-forming EW(Hα)')
    
    # (3) BPT diagram
    ax[0,2].set_title('Rejuvenation BPT diagram')
    ax[0,2].plot(np.ravel(x), np.ravel(y), '.', alpha = 0.5, markersize = 2)
    #ax[0,2].plot(np.ravel(np.ma.array(x, mask = (rej_mask))), np.ravel(np.ma.array(y, mask = (rej_mask))), '.', color='red', markersize=3)
    ax[0,2].plot(np.linspace(-1.4, 0, 50), 0.61/(np.linspace(-1.4, 0, 50)-0.05) + 1.30, '--', color = 'black')
    ax[0,2].set_xlabel('log(NII/Hα)')
    ax[0,2].set_ylabel('log(OIII/Hβ)')
    ax[0,2].set_xlim(-1.4, 0.3)
    ax[0,2].set_ylim(-1.5, 1.3)
    
    # (4) MZ relation
    M, mask_M = pips.sigma()
    M = np.ma.array(M, mask = (Ha_mask | bpt_mask)) # Surface mass density
    R = np.ma.array(maps.ellcoo(), mask = (Ha_mask | bpt_mask)) # Effective radius
    SFR = np.ma.array(maps.SFR(), mask = (Ha_mask | bpt_mask | mask_M)) # Star-formation rate
    Z, mask_Z = maps.metallicity() # Metallicity
    Z = np.ma.array(Z, mask = (Ha_mask | bpt_mask | mask_M))
    
    M_rej = np.ma.array(M, mask = rej_mask)
    Z_rej = np.ma.array(Z, mask = rej_mask)
    R_rej = np.ma.array(R, mask = rej_mask)
    SFR_rej = np.ma.array(SFR, mask = rej_mask)
    
    ax[1,0].set_title('M-Z relation')
    ax[1,0].plot(np.ravel(M), np.ravel(Z), '.', label = 'Star-forming', alpha = 0.3)
    #try:
        #ax[1,0].plot(np.ravel(M_rej), np.ravel(Z_rej), '.', label = 'Star-forming', alpha=0.7)
    #except: pass
    ax[1,0].plot(X, Y, linewidth = 1, label = 'B&B et al. 2016', linestyle = '--', color = 'black')
    ax[1,0].set_xlabel('Stellar mass [M$_\odot$/pc$^2$]')
    ax[1,0].set_ylabel('12 + log(O/H)')
    ax[1,0].legend()
    
    # (5) The metallicity gradient & (6) sSFR profile
    try:
        r, met, met_err, sfr, sfr_err = MG(gal)
        slope, intercept, r_value, p_value, std_err = linregress(r, met)
        rho = spearmanr(r, met)[0]
    
        ax[1,1].set_title('Z gradient and profile')
        s  = ax[1,1].scatter(r, met, c = sfr, cmap = 'viridis', marker = 'o', s=50, edgecolors='k', linewidths=1)
        eb = ax[1,1].errorbar(r, met, yerr = met_err, fmt = 'none', capsize = 5, ecolor = 'gray', lw = 1)
        cbar = fig.colorbar(s, ax=ax[1,1], label = 'sfr [log(M$\odot$/yr/pc$^2$)]')
        ax[1,1].plot(r, slope*r + intercept, color = 'black', ls = '--', label = 'Slope = %.2f'%slope)
        ax[1,1].plot(np.ravel(R), np.ravel(Z), '.', label = 'Star-forming', markersize = 3, alpha=0.7)
        ax[1,1].set_xlabel('r [$R_e$]')
        ax[1,1].set_ylabel('12 + log(O/H)')
        ax[1,1].legend()
        
        ax[1,2].set_title('SFR profile')
        s  = ax[1,2].scatter(r, sfr, c = met, cmap = 'viridis', marker = 'o', s=50, edgecolors='k', linewidths=1)
        eb = ax[1,2].errorbar(r, sfr, yerr = sfr_err, fmt = 'none', capsize = 5, ecolor = 'gray', lw = 1)
        cbar = fig.colorbar(s, ax=ax[1,2], label = '12 + log(O/H)]')
        #ax[1,2].plot(r, slope*r + intercept, color = 'red', ls = '--', label = 'Slope = %.2f'%slope)
        ax[1,2].plot(np.ravel(R), np.ravel(SFR), '.', label = 'Star-forming', markersize = 3, alpha=0.7)
        ax[1,2].set_xlabel('r [$R_e$]')
        #ax[1,2].set_ylabel('sfr [log(M$\odot$/yr/pc$^2$)]')
        ax[1,2].legend()
    
    except: pass
    
    plt.savefig('/Users/txl/Desktop/MaNGA/spatial/info/int/%s.png'%gal, bbox_inches = 'tight', dpi = 300)
    plt.show()