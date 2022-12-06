# This script is for training on all apertures of the preprocessed data. 
# If you want to visualise results for a single aperture use Training.ipynb notebook.
# If something does not work consider using Training.ipynb notebook to troubleshoot.

# Importing packages and functions
from scipy import interpolate
import pickle 
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import torch
import torch.nn as nn
from scipy import interpolate
from torch.utils.data import DataLoader
from functions import shift
from functions import doppler_shift
from functions import logconv
from functions import nonLogconv
from functions import EarlyStopping
from hyperparameters import hyperparams
from dataloader import dataloader

print('Loading Data...')

# Loading preprocessed data
pkl_file = open('../preproc/preproc_wave.pkl', 'rb')
preproc_wave = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../preproc/preproc_flux.pkl', 'rb')
preproc_flux = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../preproc/preproc_airmass.pkl', 'rb')
Airmass = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../preproc/preproc_berv.pkl', 'rb')
Berv = pickle.load(pkl_file)
pkl_file.close()

# Collecting dimension parameters
n_apertures = preproc_flux.shape[1] # Number of apertures in training data
P = preproc_flux.shape[2] # Pixels in each aperture

# Loading molecfit sample for custom initialisation of decoder weights
df_data = pd.read_csv('../molecfit/telluric_sample_separate.txt', delimiter = ' ', skiprows = 0,header=0)
mol_O2 = df_data['tel_O2_1']
wave_mol_O2 = df_data['wl_O2_1']
mol_H2O = df_data['tel_H2O_1']
wave_mol_H2O = df_data['wl_H2O_1']

# Converting molecfit wave to Ångstrøm
wave_mol_H2O = wave_mol_H2O* 10
wave_mol_O2 = wave_mol_O2* 10

print('Finished Loading Data\n')

# Initializing master array with dimensions:[69,3,4096]  i.e.  [aperture, (wave,H2O,O2), Pixels]
# Results are saved in this array
AE_tellurics = np.zeros([n_apertures,3,P])

# Training all apertures
print('Starting training on all apertures')
print('Training is faster on early apertures, which contain less telluric signal\n')

for order in range(0,preproc_flux.shape[1]):


    # Gathering suggested hyperparameters for the aperture
    epochs,learning_rate,n_dim= hyperparams(order)
    earlyStopping = EarlyStopping(patience=50)
    
    # Optionally change number of epochs
    epochs=epochs*2
    
    # Setting up dataloader. 
    validation = False # Decide wether to use a validation set
    batch_size,train,val,dataloader_train,dataloader_val,wave,berv_ref_init_s = dataloader(order,preproc_wave,preproc_flux,Berv,validation)
        
    # Determining learning rate based on amount of training data
    # ------------------------
    learning_rate = learning_rate / (838/batch_size)
    # ------------------------
        
    # =================================== Initializing decoder weights ============================
    # Using custom initialisation of decoder H2O and O2 weights from Molecfit sample.
    # This ensures faster convergence.
    
    # Log conversion of molecfit
    mol_o2 = logconv(mol_O2)
    mol_h2o = logconv(mol_H2O)
    
    # Interpolate to AE wave grid
    f = interpolate.interp1d(wave_mol_H2O, mol_h2o)
    mol_h2o = f(wave)
    
    f = interpolate.interp1d(wave_mol_O2, mol_o2)
    mol_o2 = f(wave)
    
    # Upscaling init Molecfit for faster convergence  
    mol_h2o = mol_h2o*5-4  
    mol_o2=mol_o2 *1.33-0.33  
    
    # Defining init O2 and H2O endmember
    O2 = mol_o2 -1
    H2O = mol_h2o -1
    
    # Defining init S endmember as observation from training set with least telluric contamination. 
    s = train[np.argmax(np.mean(train, axis=1))]
    
    # Removing Oxygen lines from init S for faster training  
    Airmass_frac = Airmass[np.argmax(np.mean(train, axis=1))] / np.max(Airmass)
    s = s-O2*Airmass_frac*1
    
    # Choosing Init scheme based on number of latent dimensions.
    init = ([s,H2O])
    if n_dim==3:
        init = ([s,H2O,O2])
    
    # Making init array readable by network.
    init = np.asarray(init).T
    init = np.float32(init)
    
    # ============================== Defining layers and model architecture  ======================
    
    num_features = len(train[0]) # Input and output layer dimension is number of bands to training data 
    class AutoEncoder(nn.Module):
        def __init__(self, latent_features):  
            super(AutoEncoder, self).__init__()
        
            self.encoder = nn.Sequential(
                nn.Linear(in_features=num_features, out_features=latent_features,bias=True), 
                nn.LeakyReLU(),   
                nn.BatchNorm1d(latent_features,affine=False,track_running_stats=True,momentum=0),
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(in_features=latent_features,out_features=num_features,bias=False) # Not using bias terms for direct interpretation
            )
     # --------------------------------------- Initializing -------------------------------------
    
            # Using custom initialization scheme for decoder weights
            self.decoder[0].weight.data = torch.from_numpy(init)
            
            # Alternativley use standard xavier init scheme
           # torch.nn.init.xavier_normal_(self.decoder[0].weight, gain=nn.init.calculate_gain('leaky_relu'))
    
            # Using xavier/glorot for encoder with optimal gain for faster learning
            torch.nn.init.xavier_normal_(self.encoder[0].weight, gain=nn.init.calculate_gain('leaky_relu'))
            
    
     # ======================================= Forward Pass =====================================
    
     # ----------------------------------------- Encoder ----------------------------------------
        def forward(self, x,ref_berv): 
            z = self.encoder(x)
            #--- ---------------- Controlling endmember abundance range -------------------
            # z are the endmember abundance weights
            # Solar endmember abundance is constrained as a constant
            z[:,0] = 1
            
            # H2O Endmember transforming to range [0,1]
            z[:,1] = z[:,1]-np.min(z[:,1].detach().numpy())        # Casting to positive values 
            z[:,1] = z[:,1] / np.max(z[:,1].detach().numpy())      # Normalizing to max value of 1
              
            # H2O Endmember transforming to range [c2,1]
            z[:,1] = z[:,1] +0.035                                 # Lower bound c2 = 1-1/(1+x). x=0.035 gives range [0.03,1]
            z[:,1] = z[:,1] / np.max(z[:,1].detach().numpy())      # Normalizing to max value of 1
            
            if n_dim==3:
                # O2 Endmember transforming to range [0,1]
                z[:,2] = z[:,2]-np.min(z[:,2].detach().numpy())    # Casting to positive values   
                z[:,2] = z[:,2] / np.max(z[:,2].detach().numpy())  # Normalizing to max value of 1
    
                # O2 Endmember transforming to range [c3,1]
                z[:,2] = z[:,2] + 2.2                              # Lower bound c3 = 1-1/(1+x). x=2.2 gives range [0.69,1]
                z[:,2] = z[:,2] / np.max(z[:,2].detach().numpy())  # Normalizing to max value of 1
                 
            # ------- Controlling endmember spectra range -------
            
            # Deciding S_top slack for clamp range. 
            S_top = 1.005
            if order==60:
                S_top=1.001
    
            # Solar
            w0 = self.decoder[0].weight[:,0].clamp(0,S_top)        # Clamping at [0,1] to interpret as continuum normalised spectrum 
            self.decoder[0].weight.data[:,0]=w0                    # Allowing clamp slightly over 1.00 to S_top in case of poor continuum normalisation
            
            # H2O
            w1 = self.decoder[0].weight[:,1].clamp(-1,0)           # Clamping at [-1,0] to interpret as absorption
            self.decoder[0].weight.data[:,1]=w1                    
            
            # O2
            if n_dim==3:
                w2 = self.decoder[0].weight[:,2].clamp(-1,0)       # Clamping at [-1,0] to interpret as absorption
                self.decoder[0].weight.data[:,2]=w2
    
     # --------------------------------------- Decoder ------------------------------------------
    
            # Reconstruction x_hat has to be constructed manually to use custom doppler shift function
            # Constructing X_hat from sum of products between latent weights (z's) and decoder weights
            # X_hat = S * w1 + H2O * w2 + O2 * W3    
            # Solar endmember is doppler shifted for each observed spectrum
            
            # H2O and O2 endmembers are identical between batch members
            # S endmember is individually doppler shifted (using defined function) towards each observed spectrum
            
            # Optionally use non-shifted solar endmembers by replacing S definition with the following line: 
            #S = self.decoder[0].weight[:,0].unsqueeze(0).repeat(batch_size, 1)   
            # This will speed up computation time a lot, but will include noise in endmembers from doppler shift of S
    
            # Expanding endmember weights to batch dimension
            S   = doppler_shift(self.decoder[0].weight[:,0].unsqueeze(0).repeat(batch_size, 1),batch_size,wave,ref_berv)

            H2O = self.decoder[0].weight[:,1].unsqueeze(0).repeat(batch_size, 1)
            
            # If n_dim = 3 include O2
            if n_dim==3: 
                O2  = self.decoder[0].weight[:,2].unsqueeze(0).repeat(batch_size, 1)
                   
            # Adding a singleton dimension to z's to perform element-wise product correctly.
            z1 =  z[:,0][:,None]
            z2 =  z[:,1][:,None]
            
            if n_dim ==3:
                z3 =  z[:,2][:,None]      
            
            # Constructing x_hat
            x_hat = S*z1 + H2O*z2   
            
            # if n_dim = 3 include O2
            if n_dim==3:
                x_hat = S*z1 + H2O*z2  + O2*z3
            
            # Collecting endmembers as decoder weights
            # Collected Solar endmember is non-shifted from initialized S. 
            decoder_weights = self.decoder[0].weight
            
            return x_hat,z,decoder_weights
            
    # Creating network and setting shape of latent space
    net = AutoEncoder(latent_features=n_dim)
    
    
    
    
    print('Training on Aperture',order,'of',preproc_flux.shape[1]-1)
    
    # ================================ Initializing training ========================================
    num_epochs = epochs # number of epochs
    
    # Initializing lists to save progress in
    t_loss=[]
    v_loss=[]
    z1_saved = []
    z2_saved = []
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training parameters
    lr=learning_rate 
    momentum = 0
    weight_decay  = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
            
    # =================================== TRAINING ===========================================
    counter = 0 # starting counter and time to track training
    start = time.time()
    
    for epoch in range(1,num_epochs+1):
        net=net.train()   # Set mode to train
        loss_train = []
        for count,data in enumerate(dataloader_train):
        
            # Define input data
            x = data
            
            # Define berv relative to init s berv
            start_id = batch_size*count     # start id of obs in batch
            stop_id  = batch_size*(count+1) # end id of obs in batch
            ref_berv = berv_ref_init_s[start_id:stop_id]
    
        # ===================forward=====================
            # clear out the gradients of variables 
            optimizer.zero_grad()    
            
            # Collect reconstruction x_hat from network
            x_hat,z,weights = net(x,ref_berv)
            
            # Define loss between reconstruction x_hat and input x
            train_loss = criterion(x_hat, x)
            
            loss_train.append(train_loss.item())

            # Saving H2O endmember abundances for plotting
            z1_saved.append(z[:,1].detach().numpy())
            
            # Save O2 abundance if n_dim = 3
            if n_dim==3:
                z2_saved.append(z[:,2].detach().numpy())
            
        # =================== backward ====================   
            train_loss.backward()
            optimizer.step()
            
        # =================== validation ==================
        if validation==True:
            loss_val = []
            with torch.no_grad():
                for data in dataloader_val:
                    net.eval()    # Set mode to evalutation

                    x_val = data 
                
                    # Setting BERVS for doppler shift of solar component
                    count+=1                        # Keeping track of batches
                    start_id = batch_size*count     # start id of obs in batch
                    stop_id  = batch_size*(count+1) # end id of obs in batch
                    if stop_id > berv_ref_init_s.shape[0]: # doppler shift function needs batches to be full
                        break                              # if bervs cannot fill a batch skip val of this batch 
                    else:
                        ref_berv = berv_ref_init_s[start_id:stop_id]
                    
                    x_hat_val,_,_  = net(x_val,ref_berv)
                    val_loss = criterion(x_hat_val, x_val)
                    loss_val.append(val_loss.item())
                
        
        # ======================== log  ===================
        t_loss.append(np.mean(loss_train))

        if validation==True:
            v_loss.append(np.mean(loss_val))
        
    
        # val_loss:    Val Loss for current batch
        # loss_val:    List of val loss pr batch for current epoch
        # v_loss:      List of val loss pr epoch (list of mean loss_val for each epoch)
    
        # train_loss:  Train Loss for current batch
        # loss_train:  List of train loss pr batch for current epoch
        # t_loss:      List of train loss pr epoch (list of mean loss_train for each epoch)
    
        
        # ======================== Early Stopping  =================== 
        if validation==True:
            if earlyStopping.step(torch.tensor(v_loss[epoch-1]),net): # val loss step at each epoch
                net = earlyStopping.bestModel # Best model is here the model where the training was stopped. 
                break  # early stop criterion is met, we can stop now
            
        
        # Printing estimated training time and update total training time used so far
        if epoch == 1:
            print(f'Estimated Training Time:  {round((time.time()-start)*num_epochs,0)} s')
    print('Total Time Elapsed for Training:',round(time.time()-start,0),'s\n')
    
    # Save weights as endmembers and save the correct spectral region
    endmember = weights.detach().numpy()
    Wave = wave
    if order in {56,59,60,61,62,65,66,67,68}:
        endmember = endmember[P:]
        Wave = wave[P:]
            
    if order==58:
        endmember = endmember[P:P*2]
        Wave = wave[P:P*2]
            
    if order<=54:
        endmember = endmember[:P]
        Wave = wave[:P]
    
    
    # Saving the endmembers
    AE_S_log = endmember[:,0] # sun
    AE_H2O_log = endmember[:,1] #  H2O
    if n_dim==3:
        AE_O2_log = endmember[:,2] #  O2
        AE_O2_log = np.clip(AE_O2_log, -1, 0)

        
    AE_H2O_log = np.clip(AE_H2O_log, -1, 0)

    # Convert to back from logspace
    AE_S = nonLogconv(AE_S_log)
    AE_H2O = nonLogconv(AE_H2O_log)
    AE_O2 = np.ones(P)
    if n_dim==3:
        AE_O2 = nonLogconv(AE_O2_log)          
    
    AE_tellurics[order,0,:] = Wave
    AE_tellurics[order,1,:] = AE_H2O
    AE_tellurics[order,2,:] = AE_O2
     
# Saving results 
print('Saving results in output directory')    
output = open('output/tellurics.pkl', 'wb')
pickle.dump(AE_tellurics, output)
output.close()  


print('Finished')
