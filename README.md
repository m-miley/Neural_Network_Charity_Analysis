# Neural_Network_Charity_Analysis

- Resources
    - Google Colab
    - charity_data.csv
    - Python 3.9.7
    - Tensorflow, Keras, Pandas

## Overview

Analysis using *deep neural networks* as an advanced classification technique to predict whether applicants for charity donation will be successful if extended endowment.  Data set includes information from over 34,000 organizations. 

## Analysis

![Screen Shot 2022-07-30 at 1 39 39 PM](https://user-images.githubusercontent.com/100544761/181937184-3eb96386-8bf2-4410-b3ad-3decf264cfb7.png)

**Preprocessing**

- Target Variable: "IS_SUCCESSFUL".  1- successful use of money. 0- unsuccessful.
- Features: ['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'INCOME_AMT', 'ASK_AMT']
- Irrelevant columns: Some columns were dropped due to their irrelevance or heavy imbalance that would confuse the model. ['EIN', 'NAME'] - I.D. columns ['STATUS', 'SPECIAL_CONSIDERATIONS'] - heavily imbalanced.

![Screen Shot 2022-07-30 at 1 49 47 PM](https://user-images.githubusercontent.com/100544761/181937504-613f99ce-129f-401d-a699-f9081488bd49.png)

Bucketing function groups rare values into "Other", thus shortening the list of values for chosen categories and resulting in less confusion or noise in model.  

![Screen Shot 2022-07-30 at 1 52 06 PM](https://user-images.githubusercontent.com/100544761/181937621-be58617d-b167-44d3-9c2e-1f6d80e7dc3b.png)

Categorical variables (strings) encoded to integers to be useful for neural network model.
