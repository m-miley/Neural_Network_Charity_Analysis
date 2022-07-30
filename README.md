# Neural_Network_Charity_Analysis

- Resources
    - Google Colab
    - charity_data.csv
    - Python 3.9.7
    - Tensorflow, Keras, Pandas

## Overview

Analysis using *deep neural networks* as an advanced classification technique to predict whether applicants for charity donation will be successful if extended endowment.  Data set includes information from over 34,000 organizations. 

## Analysis

### Data

![Screen Shot 2022-07-30 at 1 39 39 PM](https://user-images.githubusercontent.com/100544761/181937184-3eb96386-8bf2-4410-b3ad-3decf264cfb7.png)

### Preprocessing

- **Target Variable**: "IS_SUCCESSFUL".  1- successful use of money. 0- unsuccessful.
- **Features**: ['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATION', 'ASK_AMT']
- **Irrelevant columns**: Some columns were dropped due to their irrelevance or heavy imbalance that would confuse the model (optimization phase). ['EIN', 'NAME'] - I.D. columns 

![Screen Shot 2022-07-30 at 1 49 47 PM](https://user-images.githubusercontent.com/100544761/181937504-613f99ce-129f-401d-a699-f9081488bd49.png)

*Bucketing* function groups rare values into "Other", thus shortening the list of values for chosen categories and resulting in less confusion or noise in model.  ['APPLICATION_TYPE', CLASSIFICATION']

![Screen Shot 2022-07-30 at 1 52 06 PM](https://user-images.githubusercontent.com/100544761/181937621-be58617d-b167-44d3-9c2e-1f6d80e7dc3b.png)

Categorical variables (strings) *encoded* to integers to be useful for neural network model.

![Screen Shot 2022-07-30 at 2 27 55 PM](https://user-images.githubusercontent.com/100544761/181958731-3edecbb7-c924-4706-9cf4-3a92ea1c9ca9.png)
 
 Split data into train/test groups and *Scale*.

### Neural Network Model

![Screen Shot 2022-07-30 at 2 29 27 PM](https://user-images.githubusercontent.com/100544761/181960231-bd11adbe-b776-43e0-9d73-7147e8ea248d.png)

First iteration of model ran one *hidden layer* with 16 nodes and *relu activation function*.  The output layer used the *sigmoid* function to transform numerical values.  

![Screen Shot 2022-07-30 at 2 32 14 PM](https://user-images.githubusercontent.com/100544761/181962901-afa55f08-4687-4dc5-9582-575d382f2a03.png)

*Checkpoints* were established to save weights every 5th *epoch* for ease and convenience for future reference if necessary.

![Screen Shot 2022-07-30 at 2 34 01 PM](https://user-images.githubusercontent.com/100544761/181964831-e6eb7802-925c-45c5-8648-50fefc614ef1.png)

Neural Network was compiled and fit.

![Screen Shot 2022-07-30 at 2 34 13 PM](https://user-images.githubusercontent.com/100544761/181964998-0ed577a4-4cd0-4b70-b887-1f1cf251887f.png)

Then evaluated and exported to H5 file for future use.  We see here the first implementation of Neural Network Model resulted in an Accuracy of 72.6%.  Following is an attempt at *optimizing* the basic NN model for increased accuracy by strategically adjusting *hyperparameters*.

### Deep Learning Neural Network Optimization

Steps taken through many attempts at optimizing model performance are as follows:

1. Dropped two additional columns that are heavily imbalanced in their value counts: ['STATUS', 'SPECIAL_CONSIDERATIONS']
2. Bucketed three additional categories: ['ORGANIZATION','USE_CASE','AFFILIATION']
3. Increased Hidden Layer count to 2, 3, then best model.
4. Adjusted Neuron count for both layers to 32, 32, then best model.
5. Tested 'tanh' then Re-established activation function for hidden and output layers according to best model.
6. Increase epochs hyperparameter to 100
7. Implemented keras tuner and implemented the best permutation out of 60.
8. Stratify train_test_split() and adjust train/test proportionality to 80/20

![Screen Shot 2022-07-30 at 2 49 10 PM](https://user-images.githubusercontent.com/100544761/181985528-a349279a-86a2-48f3-8832-33ecc568a275.png)

Second iteration, whose only change was adding an additional layer of 16 neurons resulted in a .7% increase in accuracy.

Each additional test run of model implemented one or a couple of the strategies above.  No version of the model increased predictive accuracy above 74%.   