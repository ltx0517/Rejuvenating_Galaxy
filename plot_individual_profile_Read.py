import numpy as np
import proplot as pplt
from MyTool import reju_gal
from scipy.optimize import curve_fit

# Our Rejuvenating Galaxies
rej_massive, rej_light = reju_gal()[1:]

# Read Data
SF_spaxels_path = '/Volumes/KINGSTON/MaNGA/1D_Data/Profile/Rej_scatter/SFR/SF/'
RJ_spaxels_path = '/Volumes/KINGSTON/MaNGA/1D_Data/Profile/Rej_scatter/SFR/REJ/'
# Binning
def binning(x,y):
    x, y = np.array(x), np.array(y)
    bins = np.arange(0,2,0.1)

    digitized = np.digitize(x, bins)
    
    binned_y  = [y[digitized == i] for i in range(1, len(bins))]
    
    valid_bin = [i for i, bin_y in enumerate(binned_y) if len(bin_y)>2]
    # Find the median values for each bin
    medians = [np.median(binned_y[i]) for i in valid_bin]
    bin_center = [(bins[i] + bins[i+1]) / 2 for i in valid_bin]
    # TBD: Maybe add bootstrap method here to estimate the uncertainty of the median
    
    return bin_center, medians

def read_profile(filename):
    data = np.loadtxt(filename)
    if len(data) > 4:
        x, y = data[:, 0], data[:, 1]
        X, Y = binning(x, y)
        return X, Y
    else:
        print(f"---- File {filename} has insufficient data points. ----")
        return None, None

def rej_scatter(filename):
    data = np.loadtxt(filename)
    x, y = data[:, 0], data[:, 1]
    return x, y

def fit(x, a, b): # Fitting function to the metallicity profile
    return a*np.array(x) + b

def bootstrap(x, y, n_iter=1000):
    x, y = np.array(x), np.array(y)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    #x, y = np.array(x), np.array(y)
    n = len(x) # Total length of data
    slopes, inters = np.zeros(n_iter), np.zeros(n_iter)
    
    for i in range(n_iter):
        indices = np.random.choice(n, n, replace=True)
        x_sample, y_sample = x[indices], y[indices]
        popt, pcov = curve_fit(fit, x_sample, y_sample)
        slopes[i], inters[i] = popt[0], popt[1]
    
    slope_mean, slope_std, inter_mean, inter_std = np.mean(slopes), np.std(slopes), np.mean(inters), np.std(inters)
    
    return slope_mean, slope_std, inter_mean, inter_std

for gal in rej_massive:
    filename_sf_data = f"{SF_spaxels_path}/{gal}.txt"
    filename_rj_data = f"{RJ_spaxels_path}/{gal}.txt"

    X_sf, Y_sf = read_profile(filename_sf_data) # The star-forming spaxels profile along the radius
    X_rj, Y_rj = rej_scatter(filename_rj_data)  # The scatter of rejuvenating spaxels of galaxies
    #print(gal, X_sf, Y_sf)
    if X_sf is not None and len(X_sf)>5:
        # Do the fitting, accompanying with bootstrap
        #MG, MG_err, b, b_err = bootstrap(X_sf, Y_sf)
        
        fig, ax = pplt.subplots()
        ax.scatter(X_rj, Y_rj, s=5, color='shamrock green', label='Rejuvenating Spaxels (%i)'%len(X_rj))
        ax.scatter(X_sf, Y_sf, s=40, color='pastel blue', edgecolor='black', label='sSFR Profile')
        #ax.plot(X_sf, fit(X_sf, MG, b), '--', label='y = %.2f(%.2f) x + %.2f(%.2f)'%(MG, MG_err, b, b_err), lw=1, color='gray')
        ax.format(xlabel='Effective radius [R$_e]$', ylabel='log(sSFR) [yr$^{-1}$]', xlabelsize=12, ylabelsize=12,
                  labelsize=10, ylim=(-11, -9), title='%s'%gal)
        ax.legend(fontsize='small', ncols=1, loc='best')
        fig.savefig(f'/Volumes/KINGSTON/MaNGA/1D_Data/Profile/Individual_Profiles/{gal}.pdf', bbox_inches='tight', transparent=True, facecolor='none')
        pplt.show()