from scipy import optimize
from optimiztion import opt
from optimiztion import opt_O2
from optimiztion import error
import numpy as np
import pickle
import time
from scipy import interpolate
from continuum_normalisation import continuum_normalize


def telluric_fit(flux,wave):
    
    '''
    Function for fitting the weights of the autoencoder telluric spectrum.
    The function continuum fits the observations before fitting the telluric weights. 
    The continuum fitting is only used for finding optimal telluric weights. 
    You may you use your own continuum fitting procedure on the observations seperately from the telluric correction. 
    
    There are two cases:
    
    a) A single telluric spectrum is fitted to a single observation:
    
    Here the optimal parameters for the continuum fitting (sigma top and bottom for sigma clipping) are found
    in a simultaneous bayesian optimization scheme with the telluric weights for the specific observation.
    
    b) Mutliple telluric spectra are fitted to multiple observations:
    
    Here the optimal continuum fitting parameters are found for the first observation through bayesian optimization.
    The optimal continuum fitting parameters are then applied to all following observations.
    This assumes that all corrected observations have similar signal to noise ratio.
    The telluric weights of the observations are fitted with least squares optimization. 
    
    Input: 
    flux ---------------> Array of blaze corrected non-normalized observed spectrum flux.
    wave ---------------> Array of observed wavelength axis in units [Å] in telescope reference frame in air.
        
    Output:
    telluric spectrum --> Array of fitted telluric spectra on the wave axis of the input observations. 
    
    Input and output dimension is:  [n_obs,n_apertures,n_pixels_pr_aperture]
    For HARPS-N this is:            [n_obs,69,4096]
    ''' 
   
    # Init arrays
    flux_obs_load = flux
    wave_obs_load = wave

    n_apertures = flux_obs_load.shape[1] # Number of apertures in training data
    P = flux_obs_load.shape[2] # Pixels in each aperture
    number_of_corrections= len(flux_obs_load)

    # Initializing Arrays to save results in
    corrected_name = []
    AE_original = np.zeros([number_of_corrections,n_apertures,P])
    AE_correction_original = np.zeros([number_of_corrections,n_apertures,P])
    H2O_weights = np.zeros([number_of_corrections])
    O2_weights = np.zeros([number_of_corrections])
    AE_interp = np.zeros([number_of_corrections,n_apertures,P])

    #Loading AE tellurics (extracted endmembers from autoencoder)
    pkl_file = open('../tellurics/tellurics.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    start = time.time()
    count=0
    for obs_nr in range(number_of_corrections):

        if obs_nr ==0:
            print(f'Initializing (~ 10 sec)...\n')

        flux_obs = flux_obs_load[obs_nr]
        wave_obs = wave_obs_load[obs_nr]

        # interpolating observed spectrum to AE Wavelength Axis
        obs = np.zeros([n_apertures,P])
        for i in range(0,n_apertures): 
            f = interpolate.interp1d(wave_obs[i], flux_obs[i],bounds_error=False,fill_value=np.nan,kind='linear') 
            obs[i]  = f(data[i,0])
 
        # Bayesian optimization of optimal parameters for continuum fitting. 
        # Optimization is performed by minimizing the residual of the correction. 
        # Optimization is performed for 2 known water tellurics and a region of known O2 tellurics.
        # I.e. find three different sets of optimal [ntop,nbottom] for sigma clipping continuum normalizaiton.
        # If one wishes you can change the lines used for fitting telluric weights.
        
        if obs_nr == 0:
            trials = 150 # Number of optimization trials
            loss,ntop1,nbins1,H2O_weight1,O2_weight = \
            opt(obs_nr=obs_nr,order=53,n_trials=trials,flux=flux_obs_load,wave=wave_obs_load)
            
            loss,ntop2,nbins2,H2O_weight2,O2_weight = \
            opt(obs_nr=obs_nr,order=54,n_trials=trials,flux=flux_obs_load,wave=wave_obs_load)
            
            H2O_weight=np.mean([H2O_weight1,H2O_weight2])
            loss,ntop3,nbins3,H2O_weight,O2_weight  = \
            opt_O2(obs_nr=obs_nr,order=60,n_trials=trials,H2O_weight=H2O_weight,flux=flux_obs_load,wave=wave_obs_load)
            time_init = time.time()

        if number_of_corrections<=10:
            print(f'Correcting observation {obs_nr+1}/{number_of_corrections}')
   
        count+=1
        if number_of_corrections>10 and count==10:
            print(f'Correcting observation {obs_nr+1}/{number_of_corrections}')
            count=0
        
        # First known H2O line used for fitting telluric weights
        # === H2O - 1 ===
        ntop= ntop1 
        nbins=nbins1 
        order = 53   # Aperture
        Min = 5899.8 # Wavelength position
        Max = 5901.4 # Wavelength position
        obs = flux_obs_load[obs_nr,order]
        wave = data[order,0]
        H2O = data[order,1]
        Min_id = np.where(wave>Min)[0][0]
        Max_id = np.where(wave<Max)[0][-1]

        # Apply continuum fitting of the observed spectrum with optimal parameters found earlier
        cont_obs = continuum_normalize(wave_obs_load[obs_nr,order],obs,n_sigma=[ntop1,nbins1])

        # Locating relevant array ID for the telluric line location
        Min_id = np.where(wave_obs_load[obs_nr,order]>Min)[0][0]
        Max_id = np.where(wave_obs_load[obs_nr,order]<Max)[0][-1]
       
        # Interpolating AE spectrum to observed
        f = interpolate.interp1d(wave, H2O,bounds_error=False,fill_value=np.nan,kind='quadratic')
        AE_interp_first_H2O  = f(wave_obs_load[obs_nr,order])
    
        # Fitting known water line to find optimal abundance weight for H2O for the first line
        def H2O_fit1(w1):     
            AE_H2O = AE_interp_first_H2O*w1 + (1-w1)
            y_ae = obs[Min_id:Max_id]/AE_H2O[Min_id:Max_id]/cont_obs[Min_id:Max_id]
            rms = error(y_ae)
            return rms
        result = optimize.minimize_scalar(H2O_fit1)
        H2O_weight1 = result.x
        
        # Second known H2O line used for fitting telluric weights
        # === H2O - 2 == 
        ntop= ntop1 
        nbins=nbins1 
        order = 54
        Min = 5944.1
        Max = 5947.5
        obs = flux_obs_load[obs_nr,order]
        wave = data[order,0]
        H2O = data[order,1]
        Min_id = np.where(wave>Min)[0][0]
        Max_id = np.where(wave<Max)[0][-1]
    
        cont_obs = continuum_normalize(wave_obs_load[obs_nr,order],obs,n_sigma=[ntop2,nbins2])

        Min_id = np.where(wave_obs_load[obs_nr,order]>Min)[0][0]
        Max_id = np.where(wave_obs_load[obs_nr,order]<Max)[0][-1]
           
        # Interpolating H2O AE spectrum to observed
        f = interpolate.interp1d(wave, H2O,bounds_error=False,fill_value=np.nan,kind='quadratic')
        AE_interp_first_H2O  = f(wave_obs_load[obs_nr,order])
    
        # Fitting known water line to find optimal abundance weight for H2O for second line
        def H2O_fit1(w1):     
            AE_H2O = AE_interp_first_H2O*w1 + (1-w1)
            y_ae = obs[Min_id:Max_id]/AE_H2O[Min_id:Max_id]/cont_obs[Min_id:Max_id]
            rms = error(y_ae)
            return rms
        result = optimize.minimize_scalar(H2O_fit1)
        H2O_weight2 = result.x
    
        # Creating final H2O weights as weighted average of the two lines. 
        H2O_weight=np.average([H2O_weight1,H2O_weight2],weights=[6./10,4./10]) 
    
        # A larger region of known O2 lines for fitting telluric weights
        # ========= O2 ======
        ntop= ntop3
        nbins=nbins3 
        order = 60
        Min = 6280
        Max = 6310
        obs = flux_obs_load[obs_nr,order]
        wave = data[order,0]
        H2O = data[order,1]
        O2 = data[order,2]
        Min_id = np.where(wave>Min)[0][0]
        Max_id = np.where(wave<Max)[0][-1]
    
        Min_id = np.where(wave_obs_load[obs_nr,order]>Min)[0][0]
        Max_id = np.where(wave_obs_load[obs_nr,order]<Max)[0][-1]
    
        cont_obs = continuum_normalize(wave_obs_load[obs_nr,order][Min_id:Max_id],obs[Min_id:Max_id],n_sigma=[ntop3,nbins3])

       
        # Interpolating H2O and O2 AE spectra to observed
        f = interpolate.interp1d(wave, H2O,bounds_error=False,fill_value=np.nan,kind='quadratic')
        AE_interp_H2O  = f(wave_obs_load[obs_nr,order])
    
        f = interpolate.interp1d(wave, O2,bounds_error=False,fill_value=np.nan,kind='quadratic')
        AE_interp_O2  = f(wave_obs_load[obs_nr,order])
    
        # Fitting known O2 lines to find optimal abundance weight for O2
        # This region also containes H2O lines, so include H2O telluric spectrum with fitted weights
        w1 = H2O_weight
        def O2_fit(w2):     
            AE_H2O = AE_interp_H2O*w1 + (1-w1)
            AE_O2 = AE_interp_O2*w2 + (1-w2)
            AE_combined = AE_H2O*AE_O2
        
            y_ae = obs[Min_id:Max_id]/AE_combined[Min_id:Max_id]/cont_obs#[Min_id:Max_id]
            rms = error(y_ae)
            return rms

        result = optimize.minimize_scalar(O2_fit)
        O2_weight = result.x
        
        # Optionally overwrite fitted weighted with manual weights for testing:
#        H2O_weight = 1
#        O2_weight  = 1

        # Applying fitted weights to all orders
        Data1 =  data[:,1]*H2O_weight + (1-H2O_weight)
        Data2 =  data[:,2]*O2_weight  + (1-O2_weight)

        # Combining H2O and O2 spectra
        AE_combined = Data1*Data2
        AE_combined = np.clip(AE_combined,0,1) # If the weights are very large control that there is not negative transmission
    
        # Saving Weights
        H2O_weights[obs_nr] = H2O_weight
        O2_weights[obs_nr]  = O2_weight

        # Saving AE tellurics on its own original wave axis
        # Optionally change function to output this if the telluric spectrum without interpolation is required
        AE_original[obs_nr] = AE_combined
 
        # Interpolating combined AE to observed
        wave = data[:,0]
        for i in range(0,69): 
            f = interpolate.interp1d(wave[i], AE_combined[i],bounds_error=False,fill_value=np.nan,kind='quadratic')
            AE_interp[obs_nr,i]  = f(wave_obs[i])
    

    time_end = time.time()
    print(f'\nInitialization performed in {round(time_init-start,4)} seconds')
    print(f'\n{number_of_corrections} correction(s) performed in {round(time_end-time_init,4)} seconds')
    print(f'\nTotal time is {round(time_end-start,4)} seconds')
    print('Finished\n')
    
    return AE_interp, H2O_weights, O2_weights