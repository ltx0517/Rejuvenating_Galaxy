"""
Created on 2024/5/14
This file aims to calculate the metallicity gradient of galaxies.
"""
import numpy as np
import proplot as pplt
from scipy.optimize import curve_fit
from astropy.io import fits
#from astropy.stats import bootstrap
from MyTool import Maps, Analysis, reju_gal
from MyTool_2 import read_list_from_file
import warnings
from datetime import date
warnings.filterwarnings("ignore", category=RuntimeWarning)
today_str = date.today().strftime("%m%d")

# Read the star-forming galaxies (508 face-on star-forming galaxies, all of ba are larger than 0.7)
#gal_blue = set(read_list_from_file('/Users/txl/Desktop/MaNGA/Codes/plateifu_spiral_blue.txt'))
gal_blue = np.load('Control_SF_faceon_LTG.npy')

rej_spa = reju_gal()[0]
rej_int = ['11948-3704', '11963-9101', '12080-12705', '12624-3701', '8618-12703', '8724-12704']

#rej_spa.append('8241-12704')
#rej_spa.append('8438-9101')
#rej_spa.append('10510-12704')
#rej_spa.append('8139-1901')

with fits.open('/Users/txl/Desktop/MaNGA/Data/SDSS17Pipe3D_v3_1_1.fits') as hdu:
    data = hdu[1].data
    mass = data['log_Mass'] # The stellar mass
    galp = data['plateifu']

# Calculate for the metallicity gradient
# Read ellcoo, and obtain the medians for each radial bin. Calculate
def cal_gradient(gal):
    #gal = '11748-12701'
    maps = Maps(gal)
    mask_sf = maps.BPT_sf_mask()[2] | maps.HaEW()[1]
    ell = maps.ellcoo()
    
    R, met = [], []
    rmin, rmax, interval = 0.5, 2.0, 0.1 # Unit: Re
    for r in np.arange(rmin, rmax, interval):
        ell_mask = np.ma.masked_outside(ell, r, r+interval/2)
        ring = np.ma.getmask(ell_mask) | mask_sf
        N_ring = np.sum(~np.isnan(ring)) - np.sum(ring.compressed()) # Get the ring number
        #print(gal, N_ring, r+interval/2)
        # Check for the number of data points
        if N_ring >= 5:
            n2 = np.ma.array(maps.get_masked('NII-6585')[1], mask = ring)
            o3 = np.ma.array(maps.get_masked('OIII-5008')[1], mask = ring)
            ha = np.ma.array(maps.get_masked('Ha-6564')[1], mask = ring)
            hb = np.ma.array(maps.get_masked('Hb-4862')[1], mask = ring)
            
            # Obtain the ring metallicity by sum over all the emission line fluxes
            z = Analysis.metallicity(np.sum(ha), np.sum(hb), np.sum(n2), np.sum(o3))
            
            # Store the values
            R.append(r+interval/2)
            met.append(z)
            
    return np.array(R), np.array(met)

def fit(x, a, b): # Fitting function to the metallicity profile
    return a*np.array(x) + b

def bootstrap(x, y, n_iter=1000):
    x, y = np.array(x), np.array(y)
    n = len(x) # Total length of data
    slopes = np.zeros(n_iter)
    
    for i in range(n_iter):
        indices = np.random.choice(n, n, replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        popt, pcov = curve_fit(fit, x_sample, y_sample)
        slopes[i] = popt[0]
    
    slope_mean, slope_std = np.mean(slopes), np.std(slopes)
    
    return slope_mean, slope_std

galaxies = list(set(gal_blue) | set(rej_spa) | set(rej_int))
#galaxies = list(set(gal_blue) | set(rej_spa))
G, MG, MG_err, M = [], [], [], []
# Run over all the blue galaxies and rejuvenating galaxies
print('======= Start calculating ========')
for i, gal in enumerate(galaxies):#galaxies[:10]
    R, met = cal_gradient(gal)
    if len(met) >= 5:
        slope, slope_err = bootstrap(R, met)
        #popt, pcov = curve_fit(fit, R, met)
        #slope, slope_err = popt[0], pcov[0,0]
        #print(gal, slope_err)
        MG.append(slope)
        MG_err.append(slope_err)
        G.append(gal)
        
        # Find the stellar mass
        index = np.where(galp==gal)[0]
        M.extend(mass[index]) # log(Msun)
        #print(gal, slope, slope_err, mass[index])
    else:
        print('%s: Valid points not enough (%i)' %(gal, len(met)))
        
    if i%50 == 0: print('=== %i galaxies finished ==='%i)

# Open a file for writing
with open(f'Metallicity_gradient_blue_and_rej_v6_{today_str}.txt', 'w') as file:
    # Write the header
    file.write('G\tMG\tMG_err\tM\n')
    
    # Write the data
    for g, mg, mg_err, stellar_mass in zip(G, MG, MG_err, M):
        file.write(f'{g}\t{mg:.6f}\t{mg_err:.6f}\t{stellar_mass:.2f}\n')

# Plot part
fig, ax = pplt.subplots()
ax.hist(MG)
pplt.show()
