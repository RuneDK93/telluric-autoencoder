# telluric-autoencoder
This is a pytorch based code for $\texttt{TAU}$ (Telluric AUtoencoder), which can be used to perform quick and high accuracy telluric correction of astrophysical spectral data. See the paper for more details.

TAU is based on a constrained autoencoder structure, which learns a compressed representation of the spectral training data. From this compressed representation intepreteable componenets can be extracted. Some of these components relate to telluric absorption of light in the atmosphere of Earth. The extracted telluric spectrum can be applied to new observations to perform accurate telluric correction at low computational expense.

# Training 
The network is trained on solar observations from the HARPS-N spectrograph but can be extended to other spectrographs. 

The network can be trained for a single aperture / order using the *Master.ipynb* notebook. For training on the entire spectral range of the HARPS-N data use the *training.py* script. 

# Performing telluric correction
The extracted telluric spectrum can be used to perform telluric correction on new observations. This can be achived using the *telluric_fit* function from  *correction.py*, which is demonstrated in *AE_correction.ipynb* notebook.



