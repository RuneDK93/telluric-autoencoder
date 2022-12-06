from torchinterp1d import Interp1d
import numpy as np
import torch
import pickle 

# This file contains four helper functions and an early stopping class 
# Function 1 (shift) is a differentiable doppler shift function 
# Function 2 (doppler_shift) is a function for doppler shifting a batch of decoder weights using BERV
# Function 3 (logconv) is a function for converting a telluric transmission spectrum to network applicable log space
# Function 4 (nonLogconv) is a function for converting telluric endmembers (log space telluric transmission spectrum) back to a normal transmission spectrum
# Class 1 (EarlyStopping) is a class for early stopping of the autoencoder during training


# Function 1
def shift(flux,wave,v):
    '''
    Function for Doppler shifting a 1D torch tensor spectrum. The function expands
    on 'torchinterp1d' by allowing doppler shift controlled by input velocity.
    
    The function is similar to PyAstronomy.pyasl.dopplerShift, but this function
    does not break the computation graph and allows backpropagation through the function.
    
    Input: 
    wave -> Original wavelength axis in units [Å]
    flux -> Original spectrum flux 
    v    -> Velocity of doppler shift in units [km/s]
    
    Output:
    nflux-> New Doppler shifted flux 
    '''
    
    # Light speed in km/s
    c = 299792.458
    
    # New wavelength axis 
    wave_prime = wave * (1.0 + v / c)
    
    # Output Doppler shifted spectrum
    nflux = Interp1d()(wave,flux,wave_prime)[0]
    return nflux

# Function 2
def doppler_shift(s_weight,batch_size,wave,berv_ref_init_s):
    '''
    Function using "shift" function for shifting a batch of multiple spectra (here created from decoder weights)
    Doppler shifts are with reference to initialised solar spectrum. For each observed spectrum, 
    the S endmember is shifted towards S in the observed spectrum using the BERV of the observation.
    
    Input: 
    s_weight --------> Autoencoder weights describing solar endmember. 
    batch_size ------> Network batch size
    wave ------------> common wavegrid for training spectra 
    berv_ref_init_s -> Relative BERV between training observations and initialised solar decoder weights
    
    Output:
    shifted_s-> Doppler shifted weights 
    '''
    # creating shifted and non-shifted solar endmembers with correct dimensions
    shifted_s = torch.zeros_like(s_weight)
    #non_shifted_s = s_weight
    
    # Loop assigning individual doppler shift to solar endmember for each observation
    for i in range(batch_size):
        shifted_s[i] = shift(s_weight[i],torch.as_tensor(wave),berv_ref_init_s[i])
            
    return shifted_s

# Const is a scaling parameter found by average flux of training spetra in log space. 
# Import the scaling parameter from preproc data
pkl_file = open('../Preproc/Preproc_const.pkl', 'rb')
const = pickle.load(pkl_file)
pkl_file.close()

# Function 3
def logconv(spec):
    '''
    Function for converting a telluric transmission spectrum to network applicable log space.
    The function can be used for custom initialization of network weights with a synthetic telluric spectrum.
    '''
    spec = spec*np.exp(const) 
    spec = np.log(spec)
    spec = spec/np.max(spec)
    return spec

# Function 4
def nonLogconv(flux):
    '''
    Function for converting telluric endmembers (log space telluric transmission spectrum)
    back to a normal transmission spectrum.
    
    The function uses the constant 'const', which is a scaling parameter to bring the learned endmembers (decoder weights) 
    back on the scale of the original observations. 
    For the telluric endmembers the value of 'const' largely does not matter, as the exact scaling of these componenets
    will be determined for any new observations that need to be corrected. 
    '''
    conv   = flux
    conv = conv*const 
    conv = np.exp(conv)
    conv = conv/np.max(conv)
    return conv



# Class 1
class EarlyStopping(object):
    '''
    Class for implementing early stopping during training of neural network based on
    https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    
    The early stopping module takes in the following parameters:
    mode='min' / 'max'   - whether smaller or lager is better
    min_delta=0          - a delta that can be used to make early stopping more or less lenient when evaulating "bad" epochs
    patience='int'       - how many "bad" epochs are allowed (epoches with worse score than the current best)
    percentage='boolean' - whether the criterion is in percentage or not
    
    # MIT License
    #
    # Copyright (c) 2018 Stefano Nardo https://gist.github.com/stefanonardo
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    '''
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.bestModel = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics, curmodel):
        if self.best is None:
            self.best = metrics
            self.bestModel = curmodel
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.bestModel = curmodel
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)