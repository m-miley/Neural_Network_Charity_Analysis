# Neural_Network_Charity_Analysis

- Resources
    - Google Colab
    - charity_data.csv
    - Python 3.9.7
    - Tensorflow, Keras, Pandas

## Overview

Analysis using *deep neural networks* as an advanced classification technique to predict whether applicants for charity donation will be successful if extended endowment.  Data set includes information from over 34,000 organizations. 

## Data

![Screen Shot 2022-07-30 at 1 39 39 PM](https://user-images.githubusercontent.com/100544761/181937184-3eb96386-8bf2-4410-b3ad-3decf264cfb7.png)

## Preprocessing

- **Target Variable**: "IS_SUCCESSFUL".  1- successful use of money. 0- unsuccessful.
- **Features**: ['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATION', 'ASK_AMT']
- **Irrelevant columns**: Some columns were dropped due to their irrelevance or heavy imbalance that would confuse the model (optimization phase). ['EIN', 'NAME'] - I.D. columns 

*Bucketing* function groups rare values into "Other", thus shortening the list of values for chosen categories and resulting in less confusion or noise in model.  ['APPLICATION_TYPE', CLASSIFICATION']
![Screen Shot 2022-07-30 at 1 49 47 PM](https://user-images.githubusercontent.com/100544761/181937504-613f99ce-129f-401d-a699-f9081488bd49.png)

Categorical variables (strings) *encoded* to integers to be useful for neural network model.
![Screen Shot 2022-07-30 at 1 52 06 PM](https://user-images.githubusercontent.com/100544761/181937621-be58617d-b167-44d3-9c2e-1f6d80e7dc3b.png)
 
 Split data into train/test groups and *Scaled*.
![Screen Shot 2022-07-30 at 2 27 55 PM](https://user-images.githubusercontent.com/100544761/181958731-3edecbb7-c924-4706-9cf4-3a92ea1c9ca9.png)

## Neural Network Model

First iteration of model ran one *hidden layer* with 16 nodes and *relu activation function*.  The output layer used the *sigmoid* function to transform numerical values.  
![Screen Shot 2022-07-30 at 2 29 27 PM](https://user-images.githubusercontent.com/100544761/181960231-bd11adbe-b776-43e0-9d73-7147e8ea248d.png)

*Checkpoints* were established to save weights every 5th *epoch* for ease and convenience for future reference if necessary.
![Screen Shot 2022-07-30 at 2 32 14 PM](https://user-images.githubusercontent.com/100544761/181962901-afa55f08-4687-4dc5-9582-575d382f2a03.png)

Weights Saved.
![Screen Shot 2022-07-30 at 3 56 54 PM](https://user-images.githubusercontent.com/100544761/181995942-7ab33c31-8bd9-4472-b580-347ab3470f92.png)</br>

Neural Network was compiled and fit.
![Screen Shot 2022-07-30 at 2 34 01 PM](https://user-images.githubusercontent.com/100544761/181964831-e6eb7802-925c-45c5-8648-50fefc614ef1.png)

Then evaluated and exported to H5 file for future use.  We see here the first implementation of Neural Network Model resulted in an Accuracy of 72.6%.  
![Screen Shot 2022-07-30 at 2 34 13 PM](https://user-images.githubusercontent.com/100544761/181964998-0ed577a4-4cd0-4b70-b887-1f1cf251887f.png)

Following is an attempt at *optimizing* the basic NN model for increased accuracy by strategically adjusting *hyperparameters*.

## Deep Learning Neural Network Optimization

Steps taken through many attempts at optimizing model performance are as follows:

1. Dropped two additional columns that are heavily imbalanced in their value counts: ['STATUS', 'SPECIAL_CONSIDERATIONS']
2. Bucketed three additional categories: ['ORGANIZATION','USE_CASE','AFFILIATION']
3. Increased Hidden Layer count to 2, 3, then best model.
4. Adjusted Neuron count for both layers to 32, 32, then best model.
5. Tested 'tanh' then Re-established activation function for hidden and output layers according to best model.
6. Increase epochs hyperparameter to 100
7. Implemented *keras tuner* and implemented the best permutation out of 60.
8. Stratify train_test_split() and adjust train/test proportionality to 80/20

![Screen Shot 2022-07-30 at 2 49 10 PM](https://user-images.githubusercontent.com/100544761/181985528-a349279a-86a2-48f3-8832-33ecc568a275.png)

Second iteration, whose only change was adding an additional layer of 16 neurons resulted in a .7% increase in accuracy.
3rd - 5th iteration of model involved adjusting neuron count for each layer.  Each additional test run after that involved the adjustment of one or more of the strategies above.  No version of the model increased predictive accuracy above 74%, therefore, ultimately I was not able to increase model performance.  

**Keras Tuner**
![Screen Shot 2022-07-30 at 3 49 01 PM](https://user-images.githubusercontent.com/100544761/181995719-3a9dc48d-c7e6-4b5d-9600-2082b1f2d043.png)
![Screen Shot 2022-07-30 at 3 49 20 PM](https://user-images.githubusercontent.com/100544761/181995725-6728d359-e056-41df-a7b8-f50109050a5d.png)
*Best Model* from 60 cycles produces an accuracy score of 73%.  
![Screen Shot 2022-07-30 at 3 51 22 PM](https://user-images.githubusercontent.com/100544761/181995804-fe9a5f88-5325-4e65-96e1-b92f094e1fa4.png)

Final *Loss* and *Accuracy* for best model decided by keras tuner hyperparameter adjustments.
![Screen Shot 2022-07-30 at 3 06 35 PM](https://user-images.githubusercontent.com/100544761/181994637-aa3450ab-149f-4f9c-bef7-034204b73c96.png)



## Summary

Building Neural Network Models involve much preprocessing to get it into usable form for the model to effectively read in and make calculations.  The data in this analysis was relatively clean beforehand and required minimal transformation.  When building the basic model, it's best to start simpler and then adjust and incorporate more complex structures according to data complexity.  It's important to keep in mind that overfitting is easily accomplished with deep learning neural networks and steps taken to mitigate this possibility.  Depending on the topic, an accuracy target measurement needs to be determined before hand so that the evaluation process can have an actionable outcome.  In terms of charity donations and funding, perhaps an accuracy score between 75-90% is adequate in accepting the predictability of a model as opposed to 95-99% for healthcare analysis.

In conclusion for this analysis, an accuracy score of 75% was not reached and I would not suggest using this particular model.  If we want to continue testing this DNN model, more data should be gathered and features expanded to include more relevant data.  On the other hand, I would however recommend attempting Logistic Regression or Random Forest Classifier supervised models as they are lighter and perhaps just as accurate, respectively.  There is no terribly complex or over abundant data at hand here.  It is in tabular form, relatively clean, and prepared for simpler models such as these.  The next step should hence be in database evaluation and model exploration in search of a lighter, more accurate, and more economical model for production.
</br></br>
**Contact**</br>
mrmileyy@gmail.com</br>
[LinkedIn](https://www.linkedin.com/in/mileymarshall)