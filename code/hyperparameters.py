def hyperparams(order):
    '''
    Function for setting suggested hyperparameters. Hyperparameters are tuned for each of the 69 apertures from the HARPS-N data. 
    Differences between apertures are caused by changing levels of telluric signal in each aperture. 
    If you train on a different spectrograph, then these hyperparameters might not be optimal.
    The parameters are the suggested training parameters for the HARPS data. You can alternatively try your own parameters. 
  
    Input -----> Aperture to train on
    Output 1 --> Suggested number of epochs
    Output 2 --> Suggested learning rate
    Output 3 --> Suggested number of dimensions (number of extracted endmembers)
    '''   
    # Setting suggested dimension (number of endmembers to extract)
    n_dim=2
    if order in {27,28,29,30,41,42,43,48,49,50,51,52,57,58,59,60,61,68}:
        n_dim=3
        
    # Setting suggested number of training epochs
    if order in {53,54,55}:
        epochs = 400
        
    if order in {68}:
        epochs = 250
        
    if order in {35,36,60,63,64}:
        epochs = 200
    
    if order in {37}:
        epochs = 125
    
    if order in {49,62}:
        epochs = 100
    
    if order in {45,50}:
        epochs = 75
    
    if order in {44,51,61}:
        epochs = 50
    
    if order in {52,56,58}:
        epochs = 25
    
    if order in {59,65,66,67}:
        epochs = 20
    
    if order in range(0,34+1) or order in range(38,43+1) or order in {46,47,48,57}:
        epochs = 10
        
    # Setting suggested learning rate
    if order in {0,1,2,26,57}:
        learning_rate = 5e-2
    
    if order in range(2,17+1) or order in range(21,24+1):
        learning_rate = 5e0
        
    if order in range(18,20+1) or order in range(31,34+1) or order in {25,38,39,47,48,56,58}:
        learning_rate = 5e1
        
    if order in {30}:
        learning_rate = 10e1
       
    if order in {27,28,29,41,42,43}:
        learning_rate = 15e1
       
    if order in {35,36,37,40,44,45,46,53,54,55,63,64}:
        learning_rate = 5e2
         
    if order in {59,61,62,65,66,67,68}:
        learning_rate = 10e2
        
    if order in {60}:
        learning_rate = 12e2
        
    if order in {49,50,51,52}:
        learning_rate = 15e2
        
    return epochs,learning_rate,n_dim