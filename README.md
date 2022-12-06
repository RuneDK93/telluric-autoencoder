#  $\texttt{TAU}$: telluric-autoencoder
This is a pytorch based code for $\texttt{TAU}$ (Telluric AUtoencoder), which can be used to perform quick and high accuracy telluric correction of astrophysical spectral data. See the paper for more details.

 $\texttt{TAU}$ is based on a constrained autoencoder structure, which learns a compressed representation of the spectral training data. From this compressed representation intepreteable componenets can be extracted. Some of these components relate to telluric absorption of light in the atmosphere of Earth. The extracted telluric spectrum can be applied to new observations to perform accurate telluric correction at low computational expense.

## Performing telluric correction
Inspect the *AE_correction.ipynb* notebook for a guide on performing telluric correction. Here $\texttt{TAU}$ uses the extracted telluric spectrum on new observations to perform telluric correction. Telluric correction is performed with the *telluric_fit* function from *correction.py*.

## Training 
Use *Training.pynb* notebook for training the network on a single aperture / order. For training on the entire spectral range of the HARPS-N data use the *training.py* script. The network is trained on solar observations from the HARPS-N spectrograph but can be extended to other spectrographs. 

## Data
Data from HARPS-N used for training $\texttt{TAU}$ is not included in this repository, but can be downloaded from https://dace.unige.ch/dashboard/

A sample of 100 preprocessed observation is included in the preproc directory and can be used to test the network.

## Preprocessing data for training 
The procedure for preprocessing data prior to training the network can be seen seen in the *Preproc.ipynb*. 


