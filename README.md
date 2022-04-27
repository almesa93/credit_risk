# credit_risk

<img src="https://media.istockphoto.com/photos/stacked-credit-cards-picture-id1203763961?s=612x612" alt="drawing" width="600"/>

This project uses a Machine Learning model to detect if a client has a properly profile to grant him a credit.

To get this goal, we have a dataset with the following columns:

- SeriousDlqin2yrs (target variable) 
- RevolvingUtilizationOfUnsecuredLines
- age 
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberRealEstateLoansOrLines
- NumberOfTime60-89DaysPastDueNotWorse
- NumberOfDependents

After some investigation, the people who has a Revolving score higher than 2, are outliers. People who uses this kind of credits have, normally, economical problems.

NumberOfTime60-89DaysPastDueNotWorse and NumberOfTimes90DaysLate are variables that has a high correlation coefficient, so they could cause some disturbances in our model.

In these case, we tried with two ensembles models, Random Forest and XGBoost, which are two of the models that gives the best results, and a NN with Tensorflow and Keras, which will have even better results.

We found the following situation:

     Column                                Non-Null Count   Dtype  

     SeriousDlqin2yrs                      104805 non-null  int64  
     RevolvingUtilizationOfUnsecuredLines  104805 non-null  float64
     age                                   104805 non-null  int64  
     NumberOfTime30-59DaysPastDueNotWorse  104805 non-null  int64  
     DebtRatio                             104805 non-null  float64
     MonthlyIncome                         84024 non-null   float64
     NumberOfOpenCreditLinesAndLoans       104805 non-null  int64  
     NumberOfTimes90DaysLate               104805 non-null  int64  
     NumberRealEstateLoansOrLines          104805 non-null  int64  
     NumberOfTime60-89DaysPastDueNotWorse  104805 non-null  int64  
     NumberOfDependents                    102056 non-null  float64
 
In addition, the target value was totally unbaalanced, so it was needed to add a param in the models to fix it.
 
As you can see, there are some null values that must be preprocessed. With a SimpleImputer, we can fill the null values (mean value in case of MonthlyIncome and mode in case of NumberOfDependents) and we can make a StandardScaler to obtain better results.

The AUC score with the Random Forest Classifier model was 0.789
The AUC score with the NN Keras model was 0.842
