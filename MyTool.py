#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 12:25:18 2023

@author: txl

Relevant emission lines: Ha, Hb, OIII, NII, SII, OI
"""
from astropy.io import fits
from scipy import ndimage
#from astropy import constants as const
import numpy as np
import proplot as pplt

def reju_gal():
    s = np.loadtxt('/Users/txl/Desktop/spa_gal_v2.txt', dtype='str')
    with fits.open('/Users/txl/Desktop/MaNGA/Data/SDSS17Pipe3D_v3_1_1.fits') as hdu:
        gal_pip = hdu[1].data['plateifu']
        mass = hdu[1].data['log_Mass']
        
    reju_high, reju_low = [], []
    for gal in s:
        idx = np.where(gal_pip == gal)[0]
        if mass[idx] > 10.5:
            reju_high.append(gal)
        else:
            reju_low.append(gal)
    
    print('Total %i RJGs. %i have stellar mass < 10.5.'%(len(s), len(reju_low)))
    return s, reju_high, reju_low

class Drpall:
    
    def __init__(self, plateifu):
        self.plateifu = plateifu
        self.path = "/Users/txl/Desktop/MaNGA/Data/drpall-v3_1_1.fits"
        self.data = self.load_data()
        
    def load_data(self):
        try:
            with fits.open(self.path) as hdu:
                data = hdu[1].data
            return data
        
        except FileNotFoundError:
            print(f"Error: File '{self.path}' not found.")
            return None

        except Exception as e:
            print(f"Error: {e}")
            return None        
    
    def ba(self):
        try:
            ind = np.where(self.data['plateifu'] == self.plateifu)
            #ba = self.data['nsa_elpetro_ba'][ind][0]
            return self.data['nsa_elpetro_ba'][ind][0]
        
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def blue_cloud_plateifu(self):
        try:
            mag = self.data['nsa_elpetro_absmag']
            mag_err = self.data['nsa_elpetro_amivar']
            NUV, r = mag[:, 1], mag[:, 4]
            NUV_err, r_err = mag_err[:, 1], mag_err[:, 4]

            mask = (NUV < -100) | (r < -100) | (NUV_err < 50) | (r_err < 100) # Bad data
            mask |= ((NUV - r) >= 4)
            plateifu_blue = plateifu[~mask]
            return plateifu_blue

        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def total_stellar_mass(self):
        try:
            #ind = np.where(self.data['plateifu'] == self.plateifu)
            mass = self.data['nsa_sersic_mass']
            return np.log10(10**mass * 10/7)
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    @staticmethod
    def get_redshift(plateifu):
        try:
            with fits.open("/Users/txl/Desktop/MaNGA/Data/drpall-v3_1_1.fits") as hdu:
                data = hdu[1].data
                ind = np.where(data['plateifu'] == plateifu)
                if len(ind[0]) > 0:  # Check if plateifu is found
                    z = data['nsa_z'][ind][0]
                    return z
                else:
                    print(f"Error: Plateifu '{plateifu}' not found.")
                    return None
            #ind = np.where(self.data['plateifu'] == self.plateifu)
            #z = self.data['nsa_z'][ind][0]
            
        except Exception as e:
            print(f"Error: {e}")
            return None
            
class Maps:
    
    lines = {'OIII-5008': 16, 'NII-6585': 24, 'NII-6549': 22, 'OI-6302': 20, 'OII-3727': 0, 'OII-3729': 1, 
            'SII-6718': 25, 'SII-6732': 26, 'Ha-6564': 23, 'Hb-4862': 14}
    
    def __init__(self, plateifu):
        self.plateifu = plateifu
        self.path = self.file_path()

    def file_path(self):
        return f"/Volumes/TheUniverse/maps/manga-{self.plateifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
    
    def rejuvenated(self, adjacent_spaxel=10): # Error: Done!
        from astropy.stats import sigma_clip
        with fits.open(self.path) as hdu:
            # Read the spetra index
            hdelta  = hdu['SPECINDEX'].data[21] * hdu['SPECINDEX_CORR'].data[21]
            dn4000  = hdu['SPECINDEX'].data[44] * hdu['SPECINDEX_CORR'].data[44]
            snr_spx = hdu['SPX_SNR'].data # Spaxels quality
            
            # Read the ivar of the maps
            hd_ivar = hdu['SPECINDEX_IVAR'].data[21]
            dn_ivar = hdu['SPECINDEX_IVAR'].data[44]
            
            # To the error
            hd_err, dn_err = 1/np.sqrt(hd_ivar), 1/np.sqrt(dn_ivar)
            snr_hd, snr_dn = hdelta*np.sqrt(hd_ivar), dn4000*np.sqrt(dn_ivar)
            
            # Extension for the masks
            hd_m_extension = hdu['SPECINDEX'].header['QUALDATA']
            dn_m_extension = hdu['SPECINDEX'].header['QUALDATA']

            # The mask for the rejuvenation (and their ivar selection)
            # The qualities are dn_ivar < 7 and hd_ivar < 1.5
            #mask = (hdu[hd_m_extension].data[21]>0) | (hdu[dn_m_extension].data[44]>0) | (hd_err > 2) | (dn_err > 0.3) | (snr_spx < 3)
            mask = (hdu[hd_m_extension].data[21]>0) | (hdu[dn_m_extension].data[44]>0) | (hd_ivar < 1.5) | (dn_ivar < 7.0) | (snr_spx < 3)
            #mask = (hdu[hd_m_extension].data[21]>0) | (hdu[dn_m_extension].data[44]>0) | (np.abs(snr_hd) < 3) | (snr_dn < 3) | (snr_spx < 3)
            
            # About the rejuvenated region
            mask_rej = (hdelta > 3) | (dn4000 > 1.4) | (hdelta + 10*dn4000 - 16 > 0)
            
            # Ensure star-forming
            #mask_sf = Maps.BPT_sf_mask(self)[2] | Maps.HaEW(self)[1]
            
            # Get the final good rejuvenating spaxels
            hd = np.ma.array(hdelta, mask = mask, fill_value=np.nan)
            dn = np.ma.array(dn4000, mask = mask, fill_value=np.nan)
            
            # Doing sigma clip
            #hd, dn = sigma_clip(hd, sigma=3), sigma_clip(dn, sigma=3)
            
            """
            This part is to determine the connected spaxels.
            By using scipy.ndimage.
            """
            #print(type(~(mask_rej | mask)))
            
            #integer_mask = (~(mask_rej|mask)).astype(int)
            # Check for the grids connect or not
            labelled_array, num_feature = ndimage.measurements.label(~(mask_rej|mask))
            unique, counts = np.unique(labelled_array, return_counts = True)
            
            # Delete the non-target region spaxels
            counts, unique = np.delete(counts, 0), np.delete(unique, 0) # The empty in the maps
            
            # Mask for region selection
            want_id = np.where(counts >= adjacent_spaxel)[0]
            mask_num  = ~np.isin(labelled_array, unique[want_id])
            
            
            mask_all = mask_rej | mask | mask_num
            
        return np.ma.array(hd, mask=mask, fill_value=np.nan), np.ma.array(dn, mask=mask, fill_value=np.nan), mask_all
        
    def get_masked(self, emline, snr = 3): # This is to get the masked emission lines
        with fits.open(self.path) as hdu:
            flux = hdu['EMLINE_GFLUX'].data[Maps.lines[emline]]
            mask = hdu['EMLINE_GFLUX_MASK'].data[Maps.lines[emline]]
            ivar = hdu['EMLINE_GFLUX_IVAR'].data[Maps.lines[emline]]
            err  = 1/np.sqrt(ivar)
        
        SNR      = flux * np.sqrt(ivar)
        # Here is to obtain the two types of data
        # (1) For integrated analysis: all data
        flux1 = np.sum(np.ma.array(flux, mask = mask))
        #flux1 = np.sum(flux)
        
        # (2) For spatially analysis: bad spaxels excluded
        mask2 = mask | (flux <= 0) | (SNR < snr)
        
        flux2 = np.ma.array(flux, mask = mask2)
        err   = np.ma.array(err, mask = mask2) # Spatial error
        return flux1, flux2, mask2
    
    def spaxel_snr(self):
        with fits.open(self.path) as hdu:
            snr = hdu['spx_snr'].data
        mask = (snr <= 3) & (snr != 0.)
        
        return mask
        
    def NO_abundance(self):
        import extinction
        f = extinction.Fitzpatrick99(3.1)
        lam = np.array([3727, 3729, 6549, 6585])
        A1, A2, A3, A4 = f(lam, 1.)
        
        # Correct for the flux = flux * 10**(0.4 * extinction)
        o2_3727 = Maps.get_masked(self, 'OII-3727')[1] * 10**(0.4*A1)
        o2_3729 = Maps.get_masked(self, 'OII-3729')[1] * 10**(0.4*A2)
        n2_6549 = Maps.get_masked(self, 'NII-6549')[1] * 10**(0.4*A3)
        n2_6585 = Maps.get_masled(self, 'NII-6585')[1] * 10**(0.4*A4)
        
        N2O2 = (n2_6585 + n2_6549) / (o2_3727 + o2_3729)
        NO = 0.73*N2O2 - 0.58 # log(N/O)
        return NO
    
    def metallicity(self, mode='spatial'): # Input emission lines, output metallicity.
        # O3N2 calibration, input: OIII/Hb, NII/Ha
        n2_int, n2_spa, mask_n2 = Maps.get_masked(self, 'NII-6585')
        o3_int, o3_spa, mask_o3 = Maps.get_masked(self, 'OIII-5008')
        ha_int, ha_spa, mask_ha = Maps.get_masked(self, 'Ha-6564')
        hb_int, hb_spa, mask_hb = Maps.get_masked(self, 'Hb-4862')
        
        mask = mask_n2 | mask_o3 | mask_ha | mask_hb
        
        # For integrated analysis
        if mode == 'whole':
            O3N2 = np.log10( (o3_int/hb_int)*(ha_int/n2_int) )
            Z = 8.533 - 0.214*O3N2
            return Z, mask
        
        # For spatially analysis: (1) Distribution, (2) Gradient
        if mode == 'spatial':
            O3N2 = np.log10( (o3_spa/hb_spa )*( ha_spa/n2_spa) )
            Z    = 8.533 - 0.214 * O3N2
            return Z, mask
    '''    
    def metallicity(ha, hb, n2, o3): # Input emission lines, output metallicity.
        O3N2 = np.log10( (o3/hb)*(ha/n2) )
        return 8.533 - 0.214*O3N2
    '''
    def SFR(self, Ha, Hb, mode): # Input observed Ha, Hb, output SFR. Can be spatially or integrated.
        # Get the observed luminosity
        L_a, L_b = Maps.luminosity(self, Ha), Maps.luminosity(self, Hb)
        
        # Cardelli, Clayton & Mathis (1989) 
        # Same equation from Ashley Spindler (2018)
        La_int = L_a * (L_a/L_b/2.86)**2.36
        # Get the B/A to correct surface density
        drpall_instance = Drpall(self.plateifu)
        ba = drpall_instance.ba()
        
        if mode == 'spatial':
            # Get the spaxel area in pc
            spaxel_area_pc = Maps.spaxel_area_pc(self)
            sfr    = np.log10(La_int*ba / 10**41.1 / spaxel_area_pc) # [log(Msun/yr/pc^2)]
        if mode == 'whole':
            sfr = np.log10(La_int/10**41.1)
            
        return sfr
    '''
    def SFR(self): # Give Ha, Hb, and calculate
        """
        Calculate the star-formation rate from Kennicutt, 1998.
        SFR = 7.9 * 10^(-42) L(Ha); Ha/Hb = 2.86
        Extinction K(λ) ~ λ^(-0.7)
        L = 4πFd^2
        """
        
        # Read the emission lines
        ha_int, ha_spa, mask_ha = Maps.get_masked(self, 'Ha-6564')
        hb_int, hb_spa, mask_hb = Maps.get_masked(self, 'Hb-4862')
        
        # Get the redshift and area of one in pc
        spaxel_area_pc = Maps.spaxel_area_pc(self)
        
        # The wavelength of Ha and Hb
        lam_a, lam_b = 6563, 4861 # Unit: Angstrom
        f = (lam_b/lam_a)**(-0.7)
        
        # Obtain the luminosity
        L_a, L_b = Maps.luminosity(self, ha_spa), Maps.luminosity(self, hb_spa)
        
        # The intrinsic Ha luminosity
        x_int, x_obs = 2.86, L_a/L_b # Theoretical / Observed
        Aa = np.log10(x_obs/x_int)/np.log10(2.5)/(f-1)
        La_int = L_a/2.5**Aa
        
        # Calculate the SFR
        sfr = np.log10(7.9*10**(-42)*La_int / spaxel_area_pc) # [log(Msun/yr/pc^2)]
        
        return sfr
    '''
    def luminosity(self, flux):
        # Parameters
        H0 = 70     # Hubble constant [km/s/Mpc]
        c  = 299792 # The speed of light [km/s]
        z  = Drpall.get_redshift(self.plateifu) # The redshift
        
        #drpall_instance = Drpall(self.plateifu)
        #z = drpall_instance.get_redshift(self.plateifu)
        # Some constants we need
        # unit_factor = (3.08*10**19)**2 * 10**-17 * 10**4
        unit_factor = 3.08**2 * 10**31 # Update: 2024/4/30
        
        # Find Luminosity from Flux
        L = unit_factor*(4*3.14*flux*z**2*c**2)/H0**2
        
        return L # unit: erg/s

    def BPT_sf_mask(self): # Quality: SNR > 3
        n2_spa, mask_n2 = Maps.get_masked(self, 'NII-6585')[1:]
        o3_spa, mask_o3 = Maps.get_masked(self, 'OIII-5008')[1:]
        ha_spa, mask_ha = Maps.get_masked(self, 'Ha-6564')[1:]
        hb_spa, mask_hb = Maps.get_masked(self, 'Hb-4862')[1:]
        mask_emline = mask_n2 | mask_o3 | mask_ha | mask_hb
        
        # Star-forming mask (Only use the NII lines)
        y, x = np.log10(o3_spa/hb_spa), np.log10(n2_spa/ha_spa)
        mask_sf = ~(y < 0.61/(x-0.05) + 1.30)
        mask = mask_sf | mask_emline
        
        return x, y, mask
    
    def HaEW(self, lim=6): # Quality: SNR > 10
        with fits.open(self.path) as hdu:
            HaEW      = hdu['EMLINE_GEW'].data[23]
            HaEW_mask = hdu['EMLINE_GEW_MASK'].data[23]
            HaEW_ivar = hdu['EMLINE_GEW_IVAR'].data[23]
            HaEW_err  = 1/np.sqrt(HaEW_ivar)
            
        HaEW_snr = HaEW * np.sqrt(HaEW_ivar)
        HaEW_MaNGA = np.ma.array(HaEW, mask = HaEW_mask, fill_value = np.nan)

        HaEW_mask |= ((HaEW_snr < 10) | (HaEW <= lim))
        HaEW_sf = np.ma.array(HaEW, mask = HaEW_mask, fill_value = np.nan)
        
        return HaEW_MaNGA, HaEW_mask
    
    def spaxel_area_pc(self):
        spaxel_size = 0.5 # arcsec
        c  = 299792       # speed of light [km/s]
        H0 = 70           # Hubble constant [km/s/Mpc]
        with fits.open("/Users/txl/Desktop/MaNGA/Data/drpall-v3_1_1.fits") as hdu:
            data = hdu[1].data
        ind = np.where(data['plateifu'] == self.plateifu)
        z = data['nsa_z'][ind][0]
        # Transfer the length unit from arcsec to pc
        D = c*z/H0 # approx. distance to galaxy [Mpc]
        scale = 1 / 206265 * D * 1e6 # 1 radian = 206265 arcsec [pc / arcsec]
        spaxel_area_pc = (scale*spaxel_size)**2 # [pc^2]
        
        return spaxel_area_pc
    
    def radial_Z_sfr(self, r_min = 0.5, r_max = 2.0, interval = 0.1):
        # Read the effective radii (Re)
        with fits.open(self.path) as hdu:
            ell = hdu['SPX_ELLCOO'].data[1]
        
        # Read the stellar mass density
        sigma = Pipe3D.sigma(self)[0]
        
        # The sf or rej mask
        mask_sf  = Maps.BPT_sf_mask(self)[2] | Maps.HaEW(self)[1]
        
        # Containers
        r, met, met_err = [], [], []
        sfr, sfr_err = [], []
        
        # Calculate the radial Q (input Q should be 2D)
        for i in np.arange(r_min, r_max, interval):
            # Get the ring information
            ell_ma = np.ma.masked_outside(ell, i, i+interval/2)
            ring   = np.ma.getmask(ell_ma) | mask_sf # Give the mask for the wanted region
            N_ring = np.sum(ring.compressed())
            N_ring = np.sum(~np.isnan(ring)) - N_ring
            
            # Read the spatial emission line ([2])
            n2 = np.ma.array(Maps.get_masked(self, 'NII-6585')[1], mask = ring)
            o3 = np.ma.array(Maps.get_masked(self, 'OIII-5008')[1], mask = ring)
            ha = np.ma.array(Maps.get_masked(self, 'Ha-6564')[1], mask = ring)
            hb = np.ma.array(Maps.get_masked(self, 'Hb-4862')[1], mask = ring)

            # Metallicity part
            Z_int = Maps.metallicity(np.sum(ha), np.sum(hb), np.sum(n2), np.sum(o3))
            Z_spa = Maps.metallicity(ha, hb, n2, o3)
            # The Star-Formation Rate, unit: log(Msun/yr/pc^2)
            #sfr_int = Maps.SFR(self, np.sum(ha), np.sum(hb))
            sfr_spa = Maps.SFR(self, ha, hb) # Return the spatially-resolved SFR of the ring
            
            ssfr = sfr_spa - M_ring # A 2D array here

            # Append the results
            r.append(i+interval/2)
            met.append(Z_int)
            met_err.append(np.std(Z_spa.compressed()))
            
            
            # The ring sigma (in order to calculate sSFR)
            M_ring  = np.ma.array(sigma, mask = ring) # Unit: log[Msun/pc^2]
            M_ring_all = np.log10(10**M_ring)
            
            # The sSFR part
            #sSFR     = sfr_ring - M_ring
            #sSFR_all = sfr_ring_all - M_ring_all
            
            #print(sfr_ring_all)
            #plt.title('SFT_tot = %.1f, N_pix = %i' %(sfr_ring_all, N_ring))
            #plt.imshow(sfr_ring, origin='lower', cmap='Set2')
            #plt.colorbar()
            #plt.show()
            
            sfr.append(np.median(sSFR.compressed()))
            sfr_err.append(np.std(sSFR.compressed()))
            
        print(sfr)
        return r, met, met_err, sfr, sfr_err
    '''
    def get_optical_image(self):
        from marvin.tools.image import Image
        image = Image(self.plateifu)
        return image
    '''
    def ellcoo(self):
        with fits.open(self.path) as hdu:
            ell = hdu['spx_ellcoo'].data[1]
        return ell
    
    def velocity(self): # Read the emission line velocity
        with fits.open(self.path) as hdu:
            vel      = hdu['EMLINE_GVEL'].data[12]
            vel_ivar = hdu['EMLINE_GVEL_IVAR'].data[12]
            vel_mask = hdu['EMLINE_GVEL_MASK'].data[12]
        return vel, vel_ivar, vel_mask
    
    def velocity_dispersion(self):
        with fits.open(self.path) as hdu:
            sigma      = hdu['EMLINE_GSIGMA'].data[12]
            sigma_ivar = hdu['EMLINE_GSIGMA_IVAR'].data[12]
            sigma_mask = hdu['EMLINE_GSIGMA_MASK'].data[12]
        return sigma, sigma_ivar, sigma_mask

class Pipe3D:
    def __init__(self, plateifu):
        self.plateifu = plateifu
        self.path = self.file_path()
    
    def load_data(self):
        try:
            with fits.open(self.path) as hdu:
                data = hdu['SSP'].data
            return data
        
        except FileNotFoundError:
            print(f"Error: File '{self.path}' not found.")
            return None
        
        except Exception as e:
            print(f"Error: {e}")
            return None

    def file_path(self):
        return f"/Volumes/TheUniverse/pipe3D/manga-{self.plateifu}.Pipe3D.cube.fits.gz"
    
    def sigma(self): # Obtain the surface stellar mass density
        try:
            data = self.load_data()
            mass     = data[19] # stellar mass density per pixel with dust correction
            mass_err = data[20] # error in the Stellar Mass density
                
            # Mask the mass smaller than 0
            SNR  = mass / mass_err**2
            mask = (SNR <= 100) | (mass <= 0)
            mass = np.ma.array(mass, mask = mask) # Unit: log[Msun/spaxel]
            #fig, ax = pplt.subplots()
            #im=ax.imshow(mass)
            #ax.colorbar(im)
            #pplt.show()
            
            # Get the area of one spaxel
            spaxel_area_pc = Maps.spaxel_area_pc(self)
            
            # Get the B/A to correct surface density
            drpall_instance = Drpall(self.plateifu)
            ba = drpall_instance.ba()
            
            # Get the spatially-resolved mass
            sigma_star  = np.log10(10**mass*ba / spaxel_area_pc)  # Unit: log[Msun/pc^2]
            mask_final  = sigma_star > 4
            sigma_star  = np.ma.array(sigma_star, mask = mask_final)
        
            return sigma_star, (mask | (sigma_star > 4))
        
        except:
            print('%s has no data in Pipe3D.'%self.plateifu)
    
    def SFR(self):
        #with fits.open(self.path) as hdu:
        sfr = data['log_SFR_Ha']
        return sfr
        

class Analysis:
    @staticmethod
    def metallicity(Ha, Hb, N2, O3):
        O3N2 = np.log10( (O3/Hb)*(Ha/N2) )
        return 8.533 - 0.214*O3N2
    
    def binning(x, y, num, mode = 'median_boot'):
        from astropy.stats import bootstrap
        from astropy.utils import NumpyRNGContext
        # define the bin edges
        bin_edges = np.linspace(min(x), max(x), num=num)
        # bin the data and calculate median and percentiles
        bin_indices = np.digitize(x, bin_edges)
        
        # Mode 1: The median and the bootstrap error
        if mode == 'median_boot':            
            # Use bootstrap to calculate the median and the error
            med, err = [], []
            for i in range(1, len(bin_edges)):
                with NumpyRNGContext(1):
                    bootresult = bootstrap(y[bin_indices == i], bootnum = 2000, bootfunc = np.median)
                
                med.append(np.mean(bootresult))
                err.append(np.std(bootresult))
            return (bin_edges[1:] + bin_edges[:-1])/2, med, err
        
        # Mode 2: The median and the 16% and 84% percentile
        if mode == 'median_percentile':
            bin_medians = [np.median(y[bin_indices == i]) for i in range(1, len(bin_edges))]
            bin_percentiles16 = [np.percentile(y[bin_indices == i], 16) for i in range(1, len(bin_edges))]
            bin_percentiles84 = [np.percentile(y[bin_indices == i], 84) for i in range(1, len(bin_edges))]
            upper_lim = np.array(bin_medians) - np.array(bin_percentiles16)
            lower_lim = np.array(bin_percentiles84) - np.array(bin_medians)
            return (bin_edges[1:] + bin_edges[:-1])/2, bin_medians, upper_lim, lower_lim
        
        def sfr(Ha, Hb):
            return None
        
        @staticmethod
        def luminosity(flux, redshift):
            H0 = 70     # Hubble constant [km/s/Mpc]
            c  = 299792 # The speed of light [km/s]
            # Unit of flux: 10^(-17) erg/s/cm^2 at certain wavelength
            unit_factor = 3.08**2 * 10**31 # Update: 2024/4/30
            # Find Luminosity from Flux
            L = unit_factor*(4*3.14*flux*redshift**2*c**2)/H0**2
            return L # unit: erg/s


# Test here
'''
gal = '8978-12705'
maps = Maps(gal)
#pips = Pipe3D(gal)
#drp = Drpall(gal)
snr3, snr0 = maps.spaxel_snr()


print(drp.ba())
met = maps.metallicity()[0]
Ha = maps.get_masked('Ha-6564')[1]
Hb = maps.get_masked('Hb-4862')[1]
#z = drp.get_redshift(gal)
#La = maps.luminosity(Ha)

hd, dn, mask_rej = maps.rejuvenated()
ha, mask_ha = maps.HaEW(lim=6)
mask_bpt = maps.BPT_sf_mask()
mask_sf = mask_ha | mask_bpt

sigma = pips.sigma()[0]
#r, met, met_err, sfr, sfr_err = maps.radial_Z_sfr()
sfr = maps.SFR(Ha, Hb, mode='spatial')
#ssfr = sfr - sigma
#r, met, met_err, sfr, sfr_err = maps.radial_Z_sfr()
#Z = Analysis.metallicity(maps.get_masked('Ha-6564')[1], maps.get_masked('Hb-4862')[1], maps.get_masked('NII-6585')[1], maps.get_masked('OIII-5008')[1])

# Plot part
fig, ax = pplt.subplots(dpi=1000)
im = ax.imshow(met[0], origin='lower', cmap='IceFire')
ax.colorbar(im, loc='ul', width='0.5em', length=15)
#ax.scatter(35, 20, marker='x', color='r', s=30, lw=1)
#plt.imshow(np.ma.array(ssfr, mask = ), origin='lower', cmap='Set2')
ax.format(title='%s' %gal)
pplt.show()
'''