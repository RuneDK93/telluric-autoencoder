import numpy as np
from torch.utils.data import DataLoader

def dataloader(order,preproc_wave,preproc_flux,Berv,validation):
    '''
    Function for setting up the datloader for the given aperture to train on. 
    For many apertures the telluric signal is weak To aid the encoder in finding the correct abundance we can stack strong
    telluric apertures on the training region. Aperture 54 = strong H2O, Aperture 60 = strong O2.
    These strong telluric apertures are not saved after training and are only used to guide encoder in finding abundance. 
    
    Input 1 (order) -------------> Aperture to train on
    Input 2 (preproc_wave) ------> Preprocessed wave
    Input 3 (preproc_flux)-------> Preprocessed flux
    Input 4 (Berv)---------------> BERV of the preproc data
    Input 5 (validation)---------> validation=TRUE or FALSE. Choose wether to include validation in training.  

    Output 1 (batch_size) -------> Batch size (default batch size is the size of the training data)
    Output 2 (train) ------------> Training set (default split is 75/25)
    Output 3 (val)---------------> Validation set (default split is 75/25)
    Output 4 (dataloader_train)--> Dataloader train
    Output 5 (dataloader_val)----> Dataloader validation
    Output 6 (wave)--------------> Common wave grid for training aperture
    Output 7 (berv_ref_init_s)---> Relative BERV to the solar observation in the training set used to initialize the solar
                                   decoder weights 
    '''   
    # Collecting flux and wavelength for the aperture to train on
    flux = preproc_flux[:,order]
    wave = preproc_wave[order] 
    
    
    # Saving apertures with powerful tellurics to help train weaker apertures
    flux54 = preproc_flux[:,54]
    wave54 = preproc_wave[54]
    flux60 = preproc_flux[:,60]
    wave60 = preproc_wave[60]
    flux68 = preproc_flux[:,68]
    wave68 = preproc_wave[68]

    # Stacking relevant strong telluric apertures to aid training
    if order in {56,59,60,61,62,65,66,67,68}:
        flux = np.hstack((flux54,preproc_flux[:,order]))
        wave = np.hstack((wave54,preproc_wave[order]))
    
    if order==58:
        flux = np.hstack((flux54,preproc_flux[:,order],flux60))
        wave = np.hstack((wave54,preproc_wave[order],wave60))  
        
    if order in {27,28,29,30,41,42,43,48,49,50,51,52}:
        flux = np.hstack((preproc_flux[:,order],flux54,flux60))
        wave = np.hstack((preproc_wave[order],wave54,wave60))
        
    if order in range(0,26+1) or order in range(31,40+1) or order in range(44,47+1):
        flux = np.hstack((preproc_flux[:,order],flux54))
        wave = np.hstack((preproc_wave[order],wave54))
    
    if validation==True:
             
        # Setting batch size. 
        # Large batch size is recommended to better learn abundances. 
        # Here setting batch size as 1/4 of the preproc data.
        batch_size=int(len(flux)/4)
  
        # Train/Val is here set as a 75/25 split of the preproc data.
        # This allows large batch size, which helps find abundances and allows quicker training
        train=np.float32(flux[0:batch_size*3])  
        val = np.float32(flux[batch_size*3:])   
         # Validation dataloader
        dataloader_val = DataLoader(
          dataset=val,
          batch_size=batch_size,
          shuffle=False,
          num_workers=0)
        
        # Train dataloader
        dataloader_train = DataLoader(
          dataset=train,
          batch_size=batch_size,
          shuffle=False,
          num_workers=0)
    
    if validation==False:   
        # Return empty validation set and loader
        val =np.empty(0)
        dataloader_val=np.empty(0)
        
        # Setting batch size. 
        # Large batch size is recommended to better learn abundances. Batch size without validation set is
        # size of training data.
        batch_size=int(len(flux))
    
        # Train dataloader
        train=np.float32(flux)
        dataloader_train = DataLoader(
          dataset=train,
          batch_size=batch_size,
          shuffle=False,
          num_workers=0) 
       
    
    # Relative BERV between training observations and initialised solar decoder weights
    berv_ref_init_s = Berv - Berv[np.argmax(np.mean(train, axis=1))]
        
    return batch_size,train,val,dataloader_train,dataloader_val,wave,berv_ref_init_s