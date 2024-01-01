from astropy.io import fits
from spectres import spectres
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Decide the format of the path with difference apparatus
apparatus = {'Mac_maps_path': '/Volumes/TheUniverse/maps/',
             'Mac_pipe_path': '/Users/txl/Desktop/MaNGA/Data/Pipe3D/',
             'Windows_maps_path': 'ttt',
             'Windows_pipe_path': 'rrr'}

# =====================================
# Some relative emission lines
lines = {'OIII-5008': 16, 'NII-6585': 24, 'OI-6302': 20,
        'SII-6718': 25, 'SII-6732': 26, 'Ha-6564': 23, 'Hb-4862': 14}
# =====================================
# Functions here

def reju_gal():
    reju_int = ['7972-12703', '8240-12705', '8263-9101', '8553-12701', '9893-12704',
                '10217-9101', '11868-9102', '11962-12704', '11963-9101', '12080-12705',
                '12679-12704']
    reju_spa = ['11941-12705', '9492-12704', '8713-12701', '11748-12701', '8447-3704', '8438-12702',
                '9024-12705', '9186-6103', '11948-12704', '9866-12702', '9510-3702', '8259-6102']
    
    return reju_spa, reju_int
    
def get_masked(plateifu, emline, snr = 3, path = apparatus['Mac_maps_path']):
    # This function is to mask the bad data from the original data
    filename = path + 'manga-' + plateifu + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    
    with fits.open(filename) as hdu:
        flux = hdu['EMLINE_GFLUX'].data[lines[emline]]
        mask = hdu['EMLINE_GFLUX_MASK'].data[lines[emline]]
        ivar = hdu['EMLINE_GFLUX_IVAR'].data[lines[emline]]
        
    SNR  = flux * np.sqrt(ivar) # calculate the signal-to-noise ratio
    
    flux = np.ma.array(flux, mask = mask)
    mask |= ((flux <= 0) | (SNR < snr))
    flux = np.ma.array(flux, mask = mask)
    
    return flux, mask

def HaEW(plateifu, path = apparatus['Mac_maps_path']): # Recently star-forming signal
    filename = path + 'manga-' + plateifu + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    with fits.open(filename) as hdu:
        HaEW      = hdu['EMLINE_GEW'].data[23]
        HaEW_mask = hdu['EMLINE_GEW_MASK'].data[23]
        HaEW_ivar = hdu['EMLINE_GEW_IVAR'].data[23]
        
    HaEW_snr = HaEW * np.sqrt(HaEW_ivar)
    HaEW_mask |= (HaEW_snr < 10) | (HaEW <= 6)
    HaEW = np.ma.array(HaEW, mask = HaEW_mask)
    
    return HaEW, HaEW_mask

def sigma(plateifu, path = apparatus['Mac_pipe_path']):
    # This function is to calculate the surface mass density
    # Read the Pipe3D fits files
    with fits.open(path + 'manga-' + plateifu + '.Pipe3D.cube.fits.gz') as hdu:
        mass = hdu['SSP'].data[19] # stellar mass density per pixel with dust correction
    
    # Get the redshift and area of one spaxel
    spaxel_area_pc, redshift = spaxel_area_pc_and_redshift(plateifu)
    
    # Get the spatially-resolved mass
    sigma_star  = np.log10(10**mass / spaxel_area_pc)  # [Msun / pc^2]
    
    return sigma_star
    
def metallicity(plateifu, mode = 'spatial'):
    # O3N2 calibration
    # Read the emission lines flux and the masks
    n2, mask_n2 = get_masked(plateifu, 'NII-6585')
    o3, mask_o3 = get_masked(plateifu, 'OIII-5008')
    ha, mask_ha = get_masked(plateifu, 'Ha-6564')
    hb, mask_hb = get_masked(plateifu, 'Hb-4862')
    
    # Read the effective radius data
    file_map = 'manga-' + plateifu + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    with fits.open('/Volumes/TheUniverse/maps/' + file_map) as hdu_map:
        ell = hdu_map['spx_ellcoo'].data[1]
    ell = np.ma.array(ell, mask = (mask_n2 | mask_o3 | mask_ha | mask_hb))
    
    # Calculate the O3N2 indicator
    if mode == 'spatial':
        O3N2 = np.log10((o3/hb)*(ha/n2))
        Z    = 8.533 - 0.214 * O3N2
        
    if mode == 'whole':
        O3N2 = np.log10(( np.sum(o3)/np.sum(hb) )*( np.sum(ha)/np.sum(n2) ))
        Z    = 8.533 - 0.214 * O3N2
    
    return Z, ell
    #return Z

def SFR(plateifu, mask):
    H0 = 70     # Hubble constant [km/s/Mpc]
    c  = 299792 # speed of light [km/s]
    
    # Read the emission lines flux and the masks
    ha = np.ma.array(get_masked(plateifu, 'Ha-6564')[0], mask = mask)
    hb = np.ma.array(get_masked(plateifu, 'Hb-4862')[0], mask = mask)
    
    # Get the redshift and area of one spaxel
    spaxel_area_pc, redshift = spaxel_area_pc_and_redshift(plateifu)
    
    # Some constants we need for the luminosity calculation
    unit_factor = (3.08*10**19)**2 * 10**-17 * 10**12
    
    # Calculate the luminosity of Ha and Hb
    L_a = unit_factor*(4*3.14*np.sum(ha)*redshift**2*c**2)/H0**2
    L_b = unit_factor*(4*3.14*np.sum(hb)*redshift**2*c**2)/H0**2
    
    # The wavelength of Ha and Hb
    lam_a, lam_b = 6563, 4861 # Unit: Angstrom
    f = (lam_b/lam_a)**(-0.7)
    
    x_int, x_obs = 2.86, L_a/L_b
    
    Aa = np.log10(x_obs/x_int)/np.log10(2.5)/(f-1)
    
    # The intrinsic Ha luminosity
    La_int = L_a/2.5**Aa
    
    # Use intrinsic Ha to calculate the star-formation rate
    sfr = np.log10(7.9*10**(-42)*La_int / spaxel_area_pc) # [log(Msun/yr/pc^2)]
    
    return sfr
    
def spaxel_area_pc_and_redshift(plateifu):
    # Some constants or parameters
    spaxel_size = 0.5 # arcsec
    c  = 299792       # speed of light [km/s]
    H0 = 70           # Hubble constant [km/s/Mpc]
    
    z = get_redshift(plateifu)
    # Transfer the length unit from arcsec to pc
    D     = c * z / H0              # approx. distance to galaxy [Mpc]
    scale = 1 / 206265 * D * 1e6    # 1 radian = 206265 arcsec [pc / arcsec]
    spaxel_area_pc = (scale * spaxel_size)**2         # [pc^2]
    
    return spaxel_area_pc, z

def velocity_and_dispersion(plateifu, path = apparatus['Mac_maps_path']):
    from astropy.stats import sigma_clip
    filename = path + 'manga-' + plateifu + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    hdu  = fits.open(filename)
    
    em_vel = hdu['EMLINE_GVEL'].data[12]
    #em_vel_ivar = hdu['EMLINE_GVEL_IVAR'].data[12]
    #em_vel_mask = hdu['EMLINE_GVEL_MASK'].data[12]
    
    # Select the ok data
    em_vel[em_vel == 0] = np.nan
    em_vel[em_vel == np.nanmin(em_vel)] = np.nan
    em_vel = sigma_clip(em_vel, 3)
    
    # Get the spaxel side length
    spaxel_area = spaxel_area_pc_and_redshift(plateifu)[0]
    spaxel_length = spaxel_area ** 0.5
    
    # Calculate the velocity gradient
    # grad_x, grad_y = np.gradient(em_vel) / spaxel_length
    
    # Calculate the velocity gradient
    # em_vel_gradient = (grad_x**2 + grad_y**2)**0.5 # Unit: [km/s/pc]
    
    return em_vel, spaxel_length

def rejuvenated(plateifu, path = apparatus['Mac_maps_path']):
    filename = path + 'manga-' + plateifu + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    
    with fits.open(filename) as hdu:
    
        # Read the spetra index
        hdelta = hdu['SPECINDEX'].data[21] * hdu['SPECINDEX_CORR'].data[21]
        dn4000 = hdu['SPECINDEX'].data[44] * hdu['SPECINDEX_CORR'].data[44]
        
        # Read the ivar of the maps
        hd_ivar = hdu['SPECINDEX_IVAR'].data[21]
        dn_ivar = hdu['SPECINDEX_IVAR'].data[44]
        
        # Extension for the masks
        hd_m_extension = hdu['SPECINDEX'].header['QUALDATA']
        dn_m_extension = hdu['SPECINDEX'].header['QUALDATA']
        
        # The mask for the rejuvenation
        mask = (hdu[hd_m_extension].data[21]>0) | (hdu[dn_m_extension].data[44]>0) | (hd_ivar < 1.5)
        
        # Get the good quality hdelta and dn4000 array
        hd = np.ma.array(hdelta, mask = mask)
        dn = np.ma.array(dn4000, mask = mask)
        
        # About the rejuvenated region
        reju_condition = ((hd >= 3) | (dn >= 1.4) | (hd + 10*dn - 16 >= 0))
        
        # Get the final good rejuvenated spaxels
        hd = np.ma.array(hd, mask = reju_condition)
        dn = np.ma.array(dn, mask = reju_condition)
    
    return hd, dn, (reju_condition|mask)

def binning(x, y, num):
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext
    # define the bin edges
    bin_edges = np.linspace(min(x), max(x), num=num)

    # bin the data and calculate median and percentiles
    bin_indices = np.digitize(x, bin_edges)
    # bin_medians = [np.median(y[bin_indices == i]) for i in range(1, len(bin_edges))]
    # bin_percentiles16 = [np.percentile(y[bin_indices == i], 16) for i in range(1, len(bin_edges))]
    # bin_percentiles87 = [np.percentile(y[bin_indices == i], 87) for i in range(1, len(bin_edges))]

    # upper_lim = np.array(bin_medians) - np.array(bin_percentiles16)
    # lower_lim = np.array(bin_percentiles87) - np.array(bin_medians)
    
    # Use bootstrap to calculate the median and the error
    med, err = [], []
    for i in range(1, len(bin_edges)):
        with NumpyRNGContext(1):
            bootresult = bootstrap(y[bin_indices == i], bootnum = 1000, bootfunc = np.median)
        
        med.append(np.mean(bootresult))
        err.append(np.std(bootresult))
        
    # return (bin_edges[1:] + bin_edges[:-1])/2, bin_medians, upper_lim, lower_lim
    return (bin_edges[1:] + bin_edges[:-1])/2, med, err
    
def get_redshift(plateifu):
    with fits.open("/Users/txl/Desktop/MaNGA/Data/drpall-v3_1_1.fits") as hdu:
        data = hdu[1].data
        ind  = np.where(data['plateifu'] == plateifu)
        z = data['nsa_z'][ind][0] # Redshift of the galaxy
    return z

def radial_Z(gal, r_min = 0.5, r_max = 2.0, interval = 0.1, plot = False):
    from scipy.stats import linregress, spearmanr
    file_map = 'manga-' + gal + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    with fits.open('/Volumes/TheUniverse/maps/' + file_map) as hdu_map:
        ell = hdu_map['spx_ellcoo'].data[1]
        
    mask_sf  = BPT_sf_mask(gal)    # Get the BPT star-forming masks
    #mask_rej = rejuvenated(gal)[3] # Get the rejuvenation masks
    
    r, met, met_err, sfr = [], [], [], []
    
    # The radial metallicity part
    for i in np.arange(r_min, r_max, interval):
        ell_ma = np.ma.masked_outside(ell, i, i+interval/2)
        ring = np.ma.getmask(ell_ma) | mask_sf # Give the mask for the wanted region
        
        # Read the essential emission line
        n2 = np.ma.array(get_masked(gal, 'NII-6585')[0], mask = ring)
        o3 = np.ma.array(get_masked(gal, 'OIII-5008')[0], mask = ring)
        ha = np.ma.array(get_masked(gal, 'Ha-6564')[0], mask = ring)
        hb = np.ma.array(get_masked(gal, 'Hb-4862')[0], mask = ring)

        # Calculate the ring metallicity
        o3n2 = np.log10((o3/hb)*(ha/n2))
        O3N2 = np.log10((np.sum(o3)/np.sum(hb))*(np.sum(ha)/np.sum(n2)))
        z = 8.533 - 0.214*o3n2
        Z = 8.533 - 0.214*O3N2

        r.append(i+interval/2)
        met.append(Z)
        met_err.append(np.std(z.compressed()))
        
        # Calculate the ring SFR
        sfr.append(SFR(gal, mask = ring))
    '''
    # Exclude the bad data
    bad = ~np.logical_or(np.isnan(r), np.isnan(met), np.isnan(sfr))
    r, sfr = np.compress(bad, r), np.compress(bad, sfr)
    met, met_err = np.compress(bad, met), np.compress(bad, met_err)
    
    if len(met) >= 5:
        slope, intercept, r_value, p_value, std_err = linregress(r, met)
        rho = spearmanr(r, met)[0]
        
    if len(met) < 5: slope, r_value, rho = np.nan, np.nan, np.nan
    
    if plot == True:
        scatter = plt.scatter(r, met, c = sfr, cmap = 'viridis', marker = 'o', s=50, edgecolors='k', linewidths=1, label='slope = %.2f' %slope)
        plt.errorbar(r, met, yerr = met_err, fmt = 'none', capsize = 5, ecolor = 'gray', lw = 1)
        cbar = plt.colorbar(scatter)
        cbar.set_label('SFR [log(M$_\odot$/yr)]')
        plt.title('%s (%.2f, %.2f)' %(gal, r_value, rho))
        plt.xlabel('R/Re')
        plt.ylabel('12 + log(O/H)')
        plt.show()
    '''
    #return r, met, met_err, sfr, slope, r_value, rho
    return r, met, met_err, sfr

def weighted_avg(wave, flux, wave_min, wave_max):
    ww = (wave > wave_min) & (wave < wave_max)
    avg = np.average( flux[ww] )
    #rr = 1./np.sqrt( sum( 1./err[ww]**2  ) )
    return avg

def bp_Hd(wave, flux):
    
    wave_new = np.linspace(min(wave), max(wave), len(wave))
    flux = spectres(wave_new, wave, flux)
    wave = wave_new
    
    # Measure Lick Hd_a
    bluelim = 4041.6, 4079.75
    redlim  = 4128.5, 4161.
    feature = 4083.5, 4122.25

    if np.min(wave) < bluelim[0] and np.max(wave) > redlim[1]:
        try:
            blue_med = np.mean(bluelim)
            red_med = np.mean(redlim)
            #print(blue_med, red_med)
            
            blue_cont = weighted_avg(wave, flux, bluelim[0], bluelim[1])
            red_cont  = weighted_avg(wave, flux, redlim[0], redlim[1])
            #print(blue_cont, red_cont)
            
            dlambda = np.mean(wave[1:] - wave[:-1])
            rr = (wave >= feature[0]) & (wave <= feature[1]) # The main continuum
            
            pseudo_cont = np.interp(wave[rr], [blue_med, red_med], [blue_cont, red_cont] )
            #pseudo_cont_err = blue_cont_err * ( (wave[rr] - blue_med ) / (red_med - blue_med) )  + red_cont_err * ( (red_med - wave[rr]) / (red_med - blue_med) )
            #print(pseudo_cont)
            
            abs_feature = 1. - flux[rr] / pseudo_cont
            corr_factor = float(len(wave[rr])) / len(wave[rr])
            
            EW = np.sum(abs_feature) * dlambda * corr_factor
            #err_EW = np.sqrt( np.sum( err[rr]**2/pseudo_cont**2 ) + np.sum( flux[rr]**2 * pseudo_cont_err**2 / pseudo_cont**4  ) ) * dlambda * corr_factor
            
            return EW
       
        except:
            return -99.
    else:
        return -99., -99.
    
def bp_Dn4000(wave, flux, err=None):
    # Input: restframe wavelength, spectrum, uncertainty, mask
    # Output: Dn4000, e_Dn4000
    bluelim = 3850., 3950.
    redlim = 4000., 4100.

    if np.min(wave) < bluelim[0] and np.max(wave) > redlim[1]:
        try:
            blue_mask = (wave > bluelim[0]) & (wave < bluelim[1])
            red_mask = (wave > redlim[0]) & (wave < redlim[1])
            blue_sum = sum(flux[blue_mask]*wave[blue_mask]**2)
            red_sum = sum(flux[red_mask]*wave[red_mask]**2)
            blue_cont = blue_sum / sum(blue_mask)
            red_cont = red_sum / sum(red_mask)
            Dn = red_cont/blue_cont
            
            '''
            red_err = np.sqrt(  sum( (err[red_mask] *wave[red_mask]**2)**2 ) ) / sum(red_mask)
            blue_err = np.sqrt( sum( (err[blue_mask]*wave[blue_mask]**2)**2 ) ) / sum(blue_mask)
            eDn = Dn * np.sqrt( (red_err/red_cont)**2 + (blue_err/blue_cont)**2  )
            '''
            return Dn

        except:
            return -99.
    else:
        return -99.
    
def EW(wave, flux):
    from scipy.integrate import trapz
    import numpy as np
    # Ref: Worthey et al. (1994)
    # The range of each band, in Angstrom
    red_side  = (4128.5, 4161.00)
    blue_side = (4041.6, 4079.75)
    main_band = (4083.5, 4122.25)

    '''(1) My script'''
    # Calculate the mid-points of lam_red and lam_blue, in Angstrom
    lam_r = np.mean(red_side)
    lam_b = np.mean(blue_side)

    # Find the three bands flux
    red_idx  = np.where((wave >= red_side[0])  & (wave <= red_side[1]))
    blue_idx = np.where((wave >= blue_side[0]) & (wave <= blue_side[1]))
    main_idx = np.where((wave >= main_band[0]) & (wave <= main_band[1]))
    
    # Calculate the mean flux of red-side and blue-side
    #f_r = trapz(flux[red_idx],  wave[red_idx])  / (red_side[1] - red_side[0])
    #f_b = trapz(flux[blue_idx], wave[blue_idx]) / (blue_side[1] - blue_side[0])
    f_r, f_b = np.mean(flux[red_idx]), np.mean(flux[blue_idx])
    #print(f_r, f_b) # Smaller than PoFeng's

    # Calculate the continuum C(lambda)
    #Fc = (f_r - f_b) * (wave - lam_b) / (lam_r - lam_b) + f_b
    Fc = np.interp(wave, [lam_b, lam_r], [f_b, f_r] )
    #print(Fc)
    
    # Resample the wavelength scale
    #wave_new = np.linspace(min(wave[main_idx]), max(wave[main_idx]), len(wave[main_idx]))
    #flux_new = spectres(wave_new, wave[main_idx], flux[main_idx])

    # Calculate the equivalent width
    EW = trapz( (1 - flux[main_idx]/Fc[main_idx]) , wave[main_idx])
    #EW = trapz( (1 - flux_new/Fc[main_idx]) , wave_new)

    return EW

def BPT_sf_mask(gal, snr = 3):
    # ==== Read the lines we need ===========
    oiii, oiii_mask = get_masked(gal, 'OIII-5008', snr=snr)
    nii, nii_mask   = get_masked(gal, 'NII-6585', snr=snr)
    ha, ha_mask     = get_masked(gal, 'Ha-6564', snr=snr)
    hb, hb_mask     = get_masked(gal, 'Hb-4862', snr=snr)
    #oi, oi_mask     = get_masked(gal, 'OI-6302', snr=snr)
    
    #sii_6718, sii_6718_mask = get_masked(gal, 'SII-6718', snr = snr)
    #sii_6732, sii_6732_mask = get_masked(gal, 'SII-6732', snr = snr)
    #sii, sii_mask = (sii_6718 + sii_6732), (sii_6718_mask | sii_6732_mask)
    # =======================================
    # Mask the invalid spaxels
    bad_nii = ha_mask | oiii_mask | nii_mask | hb_mask
    #bad_sii = ha_mask | oiii_mask | sii_mask | hb_mask
    #bad_oi  = ha_mask | oiii_mask | oi_mask  | hb_mask
    mask_bad = bad_nii
    
    # Calculate the needed ratios
    y  = np.log10(oiii/hb)
    x1 = np.log10(nii/ha)
    #x2 = np.log10(sii/ha)
    #x3 = np.log10(oi/ha)
    
    # The sf seletion mask
    #mask_sf = ~((y < 0.61/(x1-0.05) + 1.30) & (y < 0.72/(x2-0.32) + 1.30) & (y < 0.73/(x3+0.59) + 1.33))
    mask_sf = ~ (y < 0.61/(x1-0.05) + 1.30)
    
    mask = mask_bad | mask_sf
    
    return mask
    
def get_optical_image(gal):
    from marvin.tools.image import Image
    image = Image(gal)
    return image

def MG(gal):
    file = '/Users/txl/Desktop/MaNGA/Data/MG/int_2/%s.fits' %gal
    with fits.open(file) as hdu:
        data = hdu[1].data
        r = data['Re']
        sfr, sfr_err = data['sfr'], data['sfr_err']
        met, met_err = data['Z'], data['Z_err']
    
    bad = ~np.logical_or(np.isnan(r), np.isnan(met))
    r, met, met_err = np.compress(bad, r), np.compress(bad, met), np.compress(bad, met_err)
    sfr, sfr_err = np.compress(bad, sfr), np.compress(bad, sfr_err)
    
    return r, met, met_err, sfr, sfr_err