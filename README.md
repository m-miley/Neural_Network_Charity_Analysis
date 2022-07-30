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

- Target Variable: "IS_SUCCESSFUL".  1- successful use of money. 0- unsuccessful.
- Features: ['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATION', 'ASK_AMT']
- Irrelevant columns: Some columns were dropped due to their irrelevance or heavy imbalance that would confuse the model (optimization phase). ['EIN', 'NAME'] - I.D. columns 

![Screen Shot 2022-07-30 at 1 49 47 PM](https://user-images.githubusercontent.com/100544761/181937504-613f99ce-129f-401d-a699-f9081488bd49.png)

Bucketing function groups rare values into "Other", thus shortening the list of values for chosen categories and resulting in less confusion or noise in model.  

![Screen Shot 2022-07-30 at 1 52 06 PM](https://user-images.githubusercontent.com/100544761/181937621-be58617d-b167-44d3-9c2e-1f6d80e7dc3b.png)

Categorical variables (strings) encoded to integers to be useful for neural network model.

![Screen Shot 2022-07-30 at 2 27 55 PM](https://user-images.githubusercontent.com/100544761/181958731-3edecbb7-c924-4706-9cf4-3a92ea1c9ca9.png)
 
 Split and Scale data into train/test groups.

### Deep Neural Network Model

![Screen Shot 2022-07-30 at 2 29 27 PM](https://user-images.githubusercontent.com/100544761/181960231-bd11adbe-b776-43e0-9d73-7147e8ea248d.png)

First iteration of model ran one hidden layer with 16 nodes and relu activation function.  The output layer used the sigmoid function to transform numerical values.  

![Screen Shot 2022-07-30 at 2 32 14 PM](https://user-images.githubusercontent.com/100544761/181962901-afa55f08-4687-4dc5-9582-575d382f2a03.png)

Checkpoints were established to save weights every 5th epoch for ease and convenience for future reference if necessary.

![Screen Shot 2022-07-30 at 2 34 01 PM](https://user-images.githubusercontent.com/100544761/181964831-e6eb7802-925c-45c5-8648-50fefc614ef1.png)

Neural Network was compiled and fit.

![Screen Shot 2022-07-30 at 2 34 13 PM](https://user-images.githubusercontent.com/100544761/181964998-0ed577a4-4cd0-4b70-b887-1f1cf251887f.png)

Then evaluated and exported to H5 file for future use.