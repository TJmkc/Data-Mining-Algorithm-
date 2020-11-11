# Data-Mining-Algorithm-
Hands on implementing Data Mining Algorithm through Python, only using Numpy package. 


    This repo aims at implementing Data Mining algorithms like Linear Regression, 
    Logistic Regression, SVM, Neural Net and etc.  I used Python as the 
    programming language and used Numpy and Pandas packages only. 


    The dataset is Appliances energy prediction data set, whihc is from UCI 
    Machine Learning Repository.  The link is https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#.

    In the implementation, i will ignore the first attribute, which is a 
    date-time variable, and i will also remove the last attribute, which is 
    a duplicate of the previous one.


    For any classification and regression task, i will use the first attribute 
    as the response variables and the remaining attributes as predictor variables.



    1. Covariance and Eigenvectors Computation.py 
    The python script will be run as :  
        Covariance and Eigenvectors Computation.py FILENAME EPS
    Here FILENAME is the name of the input CSV file, EPS the convergence 
    threshold ϵ for the eigen-vector/-value computation.
    
    
    2. Dimensional Reduction - PCA.py
    The python script will be run as : 
        Dimensional Reduction - PCA.py FILENAME ALPHA
    Here FILENAME is the datafile name, and ALPHA is the approximation threshold α.


    3. High Dimensional Data Analysis.py 
    The python script will be run as High Dimensional Data Analysis.py.
  
  
    4. Kernel PCA.py 
    The python script will be run as : 
        Kernel PCA.py FILENAME ALPHA SPREAD
    FILENAME is the datafile name, ALPHA is the approximation 
    threshold α and SPREAD is the σ^2 parameter for the Gaussian kernel.


    5. Linear Regression via QR Factorization.py
    The python script will be run as : 
        Linear Regression via QR Factorization.py FILENAME 
    where FILENAME is the input data file.
    
    
    6. Multiclass Logistic Regression.py

    For multiclass regression, I convert these into four classes as follows: 
    energy use less than or equal to 30 is class c1, 
    energy use greater than 30 but less than or equal to 50 is class c2, 
    energy use greater than 50 but less than or equal to 100 is class c3, 
    and finally energy use higher than 100 is class c4.
    
    The python script will be run as :
        Multiclass Logistic Regression.py FILENAME ETA EPS MAXITER.
    
    FILENAME is the datafile name, ETA is the step size η, 
    EPS is the convergence threshold ϵ, 
    and MAXITER is the upper bound on the number of iterations 
    when learning the weights (i.e., terminate after MAXITER even if 
    the EPS threshold has not been reached)


    7. Multiclass Classification via Neural Nets(MLP).py

    For multiclass regression, I convert these into four classes as follows: 
    energy use less than or equal to 30 is class c1, 
    energy use greater than 30 but less than or equal to 50 is class c2, 
    energy use greater than 50 but less than or equal to 100 is class c3, 
    and finally energy use higher than 100 is class c4.
    
    The python script will run as : 
    Multiclass Classification via Neural Nets(MLP).py FILENAME ETA MAXITER HIDDENSIZE NUMHIDDEN. FILENAME is the datafile name, 
      ETA is the step size η, 
      MAXITER is the number of epochs to train the model, 
      HIDDENSIZE is the size of the hidden layer, 
      and NUMHIDDEN is the number of hidden layers. 
      I also assume that all hidden layers use the same size.
    
    
    8. Support Vector Machines - Different Kernel.py

    For binary classification, 
    energy use less than or equal to 50 as the positive class (1), 
    and energy use higher than 50 as negative class (-1).
    
    The python script will run as 
    Support Vector Machines - Different Kernel.py FILENAME LOSS C EPS MAXITER KERNEL KERNEL_PARAM. 
    
    FILENAME is the datafile name, 
    LOSS is either the string "hinge" or "quadratic", 
    C is the regularization constant, EPS is the convergence threshold, 
    MAXITER is the max number of iterations to perform 
    (in case do not get convergence within EPS), 
    KERNEL is one of the strings "linear", "gaussian" or "polynomial", 
    and finally KERNEL_PARAM is either a float that represents the spread σ^2 
    for gaussian kernel, or it is a comma separated pair 
    q,c for the polynomial kernel, with q being the degree (an int) 
    and c the kernel constant (a float). 
    



    Citation:

    Luis M. Candanedo, Veronique Feldheim, Dominique Deramaix, Data driven prediction models of energy use of appliances in a low-energy house, Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97, ISSN 0378-7788, [Web Link].





