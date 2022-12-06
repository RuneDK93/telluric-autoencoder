# Importing packages
import numpy as np

def continuum_normalize(wl, fl, degree=1, n_sigma=[0.3,3.0], maxniter=50):
    """
    Fit a polynomial of a certain degree to the continuum of your spectrum and divide it out.
 
    Parameters
    ----------
    wl : 'np.array'
        1D wavelength grid of your spectrum.
    fl : 'np.array'
        Flux at each wavelength of your spectrum, with the same dimensions as wl.
    degree : 'int', optional
        Degree of the polynomial. The default is 1.
    n_sigma : 'list', optional
        Lower and upper threshold, in units of rms, to clip your data at each iteration.
        Useful to get rid of stellar absorption lines, cosmics and bad pixels.
        The default is [0.3,3.0].
    maxniter : 'int', optional
        Maximum number of iterations of your continuum fit. The default is 50.
 
    Returns
    -------
    'np.array'
        Original fl divided by the continuum fit.
 
    """
    wl = wl[~np.isnan(fl)] # remove values with nans
    fl = fl[~np.isnan(fl)]
    wl = wl[fl<1E9999999] # remove +infinities
    fl = fl[fl<1E9999999]
    wl = wl[fl>-1E9999999] # remove -infinities
    fl = fl[fl>-1E9999999]
    positions = np.ones(len(wl), dtype=bool)
    for i in range(maxniter):
        p = np.polyfit(wl[positions],fl[positions],degree)
        residuals = fl-np.polyval(p,wl)
        sigma = np.sqrt(np.nanmedian(residuals**2))
        positions_new = (residuals >= -n_sigma[0]*sigma) & (residuals <= n_sigma[1]*sigma)
        if positions_new.sum() == positions.sum(): # break code if when you are no longer clipping values
            positions = positions_new
            break
        positions = positions_new
    return np.polyval(p,wl)
