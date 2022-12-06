
#Importing packages
from continuum_normalisation import continuum_normalize
from scipy.integrate import simps
import optuna
import numpy as np
from scipy import interpolate


import pickle
optuna.logging.set_verbosity(optuna.logging.WARNING)


#Loading AE tellurics
pkl_file = open('../tellurics/tellurics.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
    

def error(y):
    """
    Function for estimating the root mean squared error (RMSE) of the continuum level. 
    The function is used to estimate the quality of telluric removal 
    (in terms of RMSE of the residuals from the correction) in a region with only telluric lines.  

    A RMSE value of zero means that the continuum is completely flat, and that no residuals remain from the telluric removal.
    A RMSE value of exactly zero is not practically possible, as the noise level in the spectra will still
    remain after telluric removal. 
    RMSE values can however be compared for telluric removal with different techniques and across different
    spectra with similar S/N.
 
    Parameters
    ----------
    y : 'np.array'
        1D array of residuals remaining from telluric correction on continuum normalized flux
        in a region with only telluric lines.
        Telluric removal should have been performed on this region. 
 
    Returns
    ---------
    mse : 'float'
        The RMSE of the continuum level. 
        A value of zero means that the continuum is completely flat, and that no residuals remain from the telluric removal.
    """

    residual = y-1 # moving the residual and continuum to be centered around 0 instead of around 1
    rmse = np.sqrt(np.mean((residual)**2)) # computing root mean squared error of residual around continuum
    return rmse


def opt(obs_nr,order,n_trials,flux,wave):
    """
    Function for finding optimal continuum fitting parameters (sigma top and bottom) by minimizing
    residual of correctionperformance in a region of known telluric lines.
    """
    # Constant parameters
    flux_obs_load = flux
    wave_obs_load = wave

    obs = flux_obs_load[obs_nr,order]
    
    if order==53:
        Min = 5899.8
        Max = 5901.4

    if order == 54:
        Min = 5944.1
        Max = 5947.5
  
    if order == 60:   
        Min = 6280 
        Max = 6310 
        
    if order ==64: 
        Min=6543.525  
        Max=6544.3


    def objective(trial): # Initiating TPE optimization trials
        ntop = trial.suggest_uniform('ntop', 0.1, 4)   # sigma top for continuum normalization
        nbins = trial.suggest_uniform('nbins', 0.1, 4) # sigma bottom for continuum normalization
        H2O_weight = trial.suggest_uniform('H2O_weight', 0, 2)
        O2_weight = trial.suggest_uniform('O2_weight', 0, 2)
        
        
        loss = continuum_fit(obs_nr,order,ntop,nbins,Min,Max,H2O_weight,O2_weight,flux_obs_load,wave_obs_load) 
        
        return loss

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Running model with best parameters
    ntop  = study.best_params.get('ntop')
    nbins = study.best_params.get('nbins')
    H2O_weight  = study.best_params.get('H2O_weight')
    O2_weight  = study.best_params.get('O2_weight')
    loss = continuum_fit(obs_nr,order,ntop,nbins,Min,Max,H2O_weight,O2_weight,flux_obs_load,wave_obs_load)
    
    return loss,ntop,nbins,H2O_weight,O2_weight

def opt_O2(obs_nr,order,n_trials,H2O_weight,flux,wave):
    """
    Function for finding optimal continuum fitting parameters (sigma top and bottom) for region of O2 absorption
    by minimizing residual of correction performance in a region of known telluric lines.
    """
    flux_obs_load = flux
    wave_obs_load = wave
    
    # Constant parameters
    obs = flux_obs_load[obs_nr,order]
    
    if order == 60:   
        Min = 6280 
        Max = 6310

    def objective(trial): 
        ntop = trial.suggest_uniform('ntop', 0.1, 4)     
        nbins = trial.suggest_uniform('nbins', 0.1, 4)
        O2_weight = trial.suggest_uniform('O2_weight', 0, 1.5)
        
        loss = continuum_fit(obs_nr,order,ntop,nbins,Min,Max,H2O_weight,O2_weight,flux_obs_load,wave_obs_load) 
        
        return loss

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Running model with best parameters
    ntop  = study.best_params.get('ntop')
    nbins = study.best_params.get('nbins')
    O2_weight  = study.best_params.get('O2_weight')
    loss = continuum_fit(obs_nr,order,ntop,nbins,Min,Max,H2O_weight,O2_weight,flux_obs_load,wave_obs_load)
    
    return loss,ntop,nbins,H2O_weight,O2_weight

def continuum_fit(obs_nr,order,ntop,nbins,Min,Max,H2O_weight,O2_weight,flux_obs_load,wave_obs_load):
    obs = flux_obs_load[obs_nr,order]
    
    wave = data[order,0]
    H2O = data[order,1] * H2O_weight + (1-H2O_weight)
    AE_telluric = H2O
    
    if order == 60:
        H2O = data[order,1] * H2O_weight + (1-H2O_weight)
        O2 = data[order,2] * O2_weight + (1-O2_weight)
        AE_telluric = H2O*O2
    
    Min_id = np.where(wave>Min)[0][0]
    Max_id = np.where(wave<Max)[0][-1]

    Min_id = np.where(wave_obs_load[obs_nr,order]>Min)[0][0]
    Max_id = np.where(wave_obs_load[obs_nr,order]<Max)[0][-1]

    cont_obs = continuum_normalize(wave_obs_load[obs_nr,order],obs,n_sigma=[ntop,nbins])

    if order ==60:
        cont_obs = continuum_normalize(wave_obs_load[obs_nr,order][Min_id:Max_id],obs[Min_id:Max_id],n_sigma=[ntop,nbins])

    
    # Interping AE to obs
    f = interpolate.interp1d(wave, AE_telluric,bounds_error=False,fill_value=np.nan,kind='quadratic')
    AE_interp  = f(wave_obs_load[obs_nr,order])
    
    if order !=60:
        y_ae = obs[Min_id:Max_id]/AE_interp[Min_id:Max_id]/cont_obs[Min_id:Max_id]  
    
    if order ==60:
        y_ae = obs[Min_id:Max_id]/AE_interp[Min_id:Max_id]/cont_obs
    
    rms = error(y_ae)
    

    return rms 
