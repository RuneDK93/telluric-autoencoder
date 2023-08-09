#  $\texttt{TAU}$: telluric-autoencoder
This is a pytorch based code for $\texttt{TAU}$ (Telluric AUtoencoder), which can be used to perform quick and high accuracy telluric correction of astrophysical spectral data. See the forthcoming paper for more details.

$\texttt{TAU}$ is based on a constrained autoencoder structure, which learns a compressed representation of the training data. The compressed representation can be used to extract interpretable components. Some of these components relate to telluric absorption of light in the atmosphere of Earth. The extracted telluric spectrum can be applied to new observations to perform accurate telluric correction at low computational expense.

The extracted telluric components are designed represent the $\mathrm{H_2O}$ and $\mathrm{O_2}$ telluric signatures in the atmosphere of Earth. The network has been trained on HARPS-N solar data, and as such the extracted components capture inherent information to the instrument, such as the point spread function. This makes the telluric components specialized for the given spectrograph (HARPS-N), but the network can also be trained on solar observations from other instruments. See the paper for a detailed evaluation on how well the extracted telluric components represent the $\mathrm{H_2O}$ and $\mathrm{O_2}$ telluric contamination in observed spectra. 

### Performing telluric correction
Inspect the *AE_correction.ipynb* notebook for a guide on performing telluric correction. Correction is performed with the *telluric_fit* function from *correction.py*.

### Training 
Use *Training.pynb* notebook for understanding the training process of $\texttt{TAU}$ by visualzing network training on a single aperture / order of the data. For training on the entire spectral range of the HARPS-N data use the *training.py* script. The network is trained on solar observations from the HARPS-N spectrograph but can be extended to other spectrographs. Before training remember to unzip the *telluric_sample_separate.zip* file in the molecfit directory.

### Preprocessing
*Preproc.ipynb* notebook demonstrates the procedure for preprocessing the raw data prior to training the network.

### Included Data
Raw data from HARPS-N is not included in this repository, but can be downloaded from https://dace.unige.ch/dashboard/

A sample of 100 preprocessed observations is included in the preproc directory and can be used to test the training procedure of $\texttt{TAU}$.

The extracted telluric spectrum from training $\texttt{TAU}$ on 838 observations is included in the tellurics directory. This extracted spectrum is used to perform correction with the *telluric_fit* function.


