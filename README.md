
# Sales Predictions Project

***

### Table of Contents

##### [Data Cleaning](#data-cleaning)

##### [Exploratory Data Analysis](#exploratory-data-analysis)

##### [Modeling Part 1](#modeling-part-1) 

##### [Feature Engineering and Modeling Part 2](#feature-engineering-and-modeling-part-2) 

##### [Interpretable Models](#interpretable-models) 

##### [Tuning Hyperparameters](#tuning-hyperparameters) 

##### [Project References](#project-references) 

***

### Background Information

> This project was inspired by the Big Mart Sales competition put on by Analytics Vidhya. The full problem statement
> can be viewed on the competition page [here](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/#LeaderBoard).
> 
> 

***

### Project Goals

> 
>1) Explore the BigMart sales dataset to uncover patterns, interesting relationships, and inconsistent or missing information. 
>2) Prepare for model building by ensuring all missing information has been handled appropriately, and all categorical information has
>   been properly encoded.
>3) Build machine learning models that can accurately predict the sales of various products across each of BigMarts 10 store locations.
>4) Derive insight from the models to answer questions such as "which of the input features have the greatest impact on sales?"
>
In the sections that follow I will explain how each component of this project worked to address the goals listed above, as well as the approach taken to do so. With that said, I hope you choose to open each notebook and follow along as you read through this document. There is a significant amount of inline explanation that takes place through the codes comments that simply cannot be fully reproduced here. 

***
***

##### [Data Cleaning](#data-cleaning)

>The code for this section can be found in the file named 01_Sales_Predictions_Data_Cleaning.ipynb

This notebook takes as an input a csv file (sales_predictions.csv) that contains the BigMart sales dataset. 

The csv file is read into a pandas DataFrame, which is then used to explore and gain familiarity with the various features and observations the dataset contains. Early exploration steps involve viewing the data types of each feature, checking for duplicated records, and reviewing the various categories within each feature to see if observations were consistently recorded. As a quick aside, this project exclusively used the built in Pandas library functions for the early data exploration steps, however much of this can be easily automated using the Pandas Profiling library, which if interested can be found [here](https://github.com/pandas-profiling/pandas-profiling). Our initial exploration revealed that the item fat content feature did contain some inconsistently recorded observations. For example, 'LF', 'low fat', and 'Low Fat' were three different inputs that intended to indicate the same thing. Since this input structure variability would lead to inaccurate category encodings that could mislead our machine learning models, all inputs were made consistent prior to proceeding. Next, the records that contained missing values were viewed in two different ways, first by performing Pandas DataFrame filtering and second by creating visualizations using the python library missingno. This review of missing values showed that 2 of the 11 features contained missing information, 'Outlet Size' and 'Item Weight'. Additionally, we gained some insight by noticing that these missing values did not appear to be entirely random, as the majority of the missing weight values came from the earliest year in the dataset, 1985 (maybe they didn't have scales back then?). This could be indicative of an error in the data collection or aggregation process, therefore in a real scenario this is the point where the data scientist should pause and reach out to the subject matter expert or individual who supplied the data to see if there is a way to obtain the correct information. In this situation I assumed the correct values could not be obtained and proceeded to fill in the missing information with the help of Scikit-Learns Simpleimputer. The missing weight values were filled in using the average item weight value, and the missing outlet size values were flagged by filling in a value of "missing". 

The result of the work outlined above is a "clean" Pandas DataFrame that is free of inconsistently recorded and missing information. This notebook concludes by exporting this clean DataFrame as a new csv file, sales_predictions_clean.csv. 


***

##### [Exploratory Data Analysis](#exploratory-data-analysis)

The code for this section can be found in the file named 02_Sales_Predictions_EDA.ipynb

This notebook takes as an input the clean csv file that is output by the 01_Sales_Predictions_Data_Cleaning.ipynb notebook.

In this section I really dive in by creating and analyzing multiple data visualizations with the intent of understanding the individual distributions of each feature and the relationships between them. A figure is worth a thousand words so for the full story, you should really check them out! 

Some key take-aways from the exploratory data analysis steps are as follows: 
- The distribution of the target variable is right (positively) skewed. Due to the underlying assumptions they make, the target will need to be transformed (e.g. box-cox or log) to a normal distribution before it could be accurately predicted using any linear model. 
- The data cleaning decision to impute the missing item weight value with mean item weight has resulted in a huge spike in the distribution at that mean value. 
- We saw that the item weight imputation decision impacted Supermarket Type3 and Grocery stores the most. For Supermarket Type3 stores, 100% of the item weight values were initially missing, and therefore after imputation were equal to the mean value. Similarly, a large portion of grocery stores had the item weight value imputed which heavily swayed the distribution such that other "in family" values would now be flagged as outliers by the normal 1.5 * IQR standard. The large impact that our data cleaning decision had on the item weight distributions for Supermarket Type3 and grocery stores was concerning, and drove us to consider what other imputation methods may have been a better fit for this dataset. 

***

##### [Modeling Part 1](#modeling-part-1)

The code for this section can be found in the file named 03_Modeling_Part_1.ipynb

This notebook takes as an input the clean csv file that is output by the 01_Sales_Predictions_Data_Cleaning.ipynb notebook.

Despite the concerns acknowleged above, in this notebook the decision is made to proceed onwards and begin creating models without further adjustments to the dataset. The rationale behind this decision is that even if we ultimately revisit and reperform data cleaning prior to final model creation, that the cross validation metrics calcuated on the models trained using this dataset will be valuable as a benchmark to compare future models to. 

To efficiently evaluate multiple model types and allow some variation in hyperparameter settings the Scikit-Learn Gridsearch function is utilized. A pipeline object feeds into the GridSearch which contains a set of column transformers, a feature selector, and an object that can vary the model type being used. The column transformer handles the final preparation steps to ensure the data is ready to feed into a machine learning model. Specifically, the numeric variables are scaled, and the categorical variables and encoded using either a one-hot or ordinal encoder depending on the variable type. The GridSearch then evaluates a variety of model types including linear regression, lasso regression, k-nearest neighbors, random forest, and bagged trees models. Additionally, the Scikit-Learn SelectKBest feature selector is used to evaluate the relative importance of each feature using an F-test. Gridsearch uses this relative feature importance information to vary the number of "best features" included in the models. This results in a diverse set of models where the model type, model specific parameters, and features used to train the models have all been varied. 

In total, the GridSearch used 5-fold cross validation to evaluate and score 7992 unique models (which means 39960 models were fit and tested within the GridSearch). The best of which was a random forest regression model that incorporated 35 features. The model parameters used 100 trees in the random forest, where each tree was constrained to a max depth of 7 levels. This model had an average root mean squared error of 1086.602 as calculated with cross validation. 

The pickle library was used to save the results of all 7992 model evaluations performed during the GridSearch into the file named gridsearch_models_1.pkl. Saving the GridSearch results in this manner allows us to easily load and view the results in the future without having to reperform the model fitting process. 

***

##### [Feature Engineering and Modeling Part 2](#feature-engineering-and-modeling-part-2) 

The code for this section can be found in the file named 04_Feature_Engineering_and_Modeling_Pt2.ipynb

This notebook takes as an input the original BigMart sales dataset (sales_predictions.csv) without any of the data cleaning work that was performed in the 01_Sales_Predictions_Data_Cleaning.ipynb notebook.

This notebook begins by taking a new approach to the data cleaning and imputation problems. Specifically, values of "0" in the item visibility feature are now considering as missing information, based on the domain specific knowledge that no items in a store could truly have zero visibility. This is addressed by replacing all values where visibility is zero with the average visibility of the other items with that same item identifier. Similarly, rather than using a global average or flagging the values into a new category as before, the missing item weight and outlet size values are imputed using the mean and mode values within the specific item identifier and outlet type categories, respectively. 

To decrease the number of categories that must be encoded, a new feature is created, "Item_Id_binned". This feature leverages the fact that the first two characters in each item identifier is always one of three values, 'FD', 'DR' or 'NC', which indicates if the particular item is 'Food', 'Drink' or 'Non-Consumable'. By using the broad Item_Id_binned category and dropping the original Item_Identifier feature, the dimensionality increase that occurs when encoding the feature is significantly reduced. Specifically, the original item identifier feature has 1559 unique values, which with a one hot encoding method would require 1558 additional dimensions to represent (a stark contrast to the 3 category feature that is replacing it) which is a significant dimensionality increase for a dataset with only 8523 observations. 

This new perspective on the item identifier feature reveals another contradiction within the data. Some observations with a "Non-Consumable" item identifier have values listed in the Item_Fat_Content feature. Since it is obviously illogical to list a fat content for something that is not food, a new category 'Not_Edible' is created within the Item_Fat_Content feature to capture these situations. 

This DataFrame with the new set of data cleaning and feature engineering decisions is exported to the file Sales_Training_Data_Cleaned_Imputed.csv for future use. 

A GridSearch is performed to evaluate various model types as described in modeling part 1. Again, pickle is used to dump the GridSearch results into a file, gridsearch_models_2.pkl, for future use. A somewhat suprising result, that despite significant changes to the input features matrix, the best model found by GridSearch has similar parameters to what was seen in part 1: a random forest regressor with 100 trees in the forest where each tree was constrained to a maximum depth of 7 levels. The model root mean squared metric as calculated using cross validation improved ever so slightly, to 1085.33. 

Next, this best model was put to the test and was evaluated using the test data set provided by Analytics Vidhya, available [here](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/#LeaderBoard). When the test data is initially downloaded it is not suitable for use as an input to a machine learning model (i.e. missing values exist and the features do not match the ones I have engineered for my training data). 

Two separate imputation strategies were used when preparing the test data. First, the exact same mean (item weight) and mode (outlet size) values that were used to imput the training data set were used to impute the test data. Second, new mean and mode values were calculated by considering the training and test data observations collectively, and then these values were used to impute. My best model was then used to make predictions using the test data prepared in each manner, and Analytics Vidhya returned RMSE scores of 1151.67 and 1155.59 respectively.

In both cases the RMSE calculated by Analytics Vidhya was higher than the average RMSE calculated using 5-fold cross validation. Also somewhat suprisingly, the test data set that was imputed using slightly more information resulted in a worse RMSE score. This slightly lower RMSE resulting from the use of the additional training data for missing value imputation could indicate outliers in the training data actually moved the imputed information further away from the true value. 

A potential third option where all missing values (both in the test and training datasets) are imputed using all the available information (test and training sets collectively) has not yet been performed. 

The test datasets were saved for future use, Sales_Test_Data_Cleaned_TrainImpute.csv and Sales_Test_Data_Cleaned_ComboImpute.csv


***

##### [Interpretable Models](#interpretable-models) 
The code for this section can be found in the file named 05_Interpretable_Modeling.ipynb

Often times there is a trade-off between model complexity and interpretability. If prediction accuracy is the only goal, then it may be true that a complex model that obfuscates the details of exactly how each input feature contributes the observed output is the best tool for the job. In this notebook, we prioritize interpretability and build some linear models where nice equations can be generated that directly link input features to weighted coefficients describing their relative importance to the calculated output. 

We once again consult our dear friend, GridSearch. Two difference sets of linear models are calculated, the difference between them being in set 1 the outlet location type and outlet type features were chosen to be encoded was nominal categorical variables. In set 2, we encoded the outlet location type and outlet type features as ordinal categorical variables. 

For both sets of models the model type was varied between linear regression, lasso regression and ridge regression. Models were considered both with and without fitting the y-intercept, as well as both with and without transforming the target vector using a box-cox transform. Additionally, models were built using everywhere from 5-all the input features. 

The resulting best model in set 1 was a Linear Regression model that incorporated 16 model features and had a calculated RMSE of 1103.81. The equation relating the input features to the sales price is shown below:  

Sales = 0.01270*(x0_Fruits and Vegetables) +  0.08816*(x1_OUT017) +  0.56196*(x1_OUT018) +  -0.6845*(x1_OUT019) +  0.98404*(x1_OUT027) +  0.11462*(x1_OUT035) +  -0.2990*(x1_OUT049) +  -0.0383*(x2_Tier 2) +  -0.7043*(x2_Tier 3) +  0.94062*(x3_Supermarket Type1) +  0.56196*(x3_Supermarket Type2) +  0.98404*(x3_Supermarket Type3) +  0.01272*(x4_Food) +  0.35044*(Outlet_Size) +  -0.0573*(2 years) +  0.00885*(4 years) 

The best model in the second set was also a Linear Regression model, this time with 13 features used, and an RMSE of again 1103.81. 
The equation relating the input features to the sales price is shown below: 

Sales = 0.01270*(x0_Fruits and Vegetables) + 0.08816*(x1_OUT017) + -1.6986*(x1_OUT018) + -0.0570*(x1_OUT019) + -2.4226*(x1_OUT027) + 0.11462*(x1_OUT035) + 0.01475*(x1_OUT049) + 0.01272*(x2_Food) + 0.03666*(Outlet_Size) + -0.0383*(Outlet_Location_Type) + 1.56817*(Outlet_Type) + -0.0573*(2 years) + 0.00885*(4 years) 

***

##### [Tuning Hyperparameters](#tuning-hyperparameters) 
The code for this section has not yet been uploaded.

***

##### [Project References](#project-references) 

This project would not have been possible without the help of numerous online resources that provided guidance, background knowledge, and helped answered questions that came up along the way. I made a great effort to keep track of these resources and have provided a list of links below. Thank you to everyone who was involved in creating and sharing this information. 

In addition to online resources, below I also list several awesome books that I kept close by throughout the process of working on this project. Each of these books has been the key that helped fill in a knowledge gap and push me passed an issue at one point or another. Thank you to the authors and all those involved in creating and sharing this material. 

**Textbook Resources:** 
1. Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurelien Geron
2. Effective Python by Brett Slatkin 
3. Python Data Science Handbook by Jake VanderPlas
4. Python for Data Analysis by Wes McKinney
5. An Introduction to Statistical Learning by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani
6. The Elements of Statistical Learning by Trevor Hastie, Robert Tibshirani and Jerome Friedman

**Online Resources:**
1. https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/
2. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
3. https://stackoverflow.com/questions/33376078/python-feature-selection-in-pipeline-how-determine-feature-names
4. https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api
5. https://github.com/scikit-learn/scikit-learn/pull/13307/files
6. https://github.com/scikit-learn/scikit-learn/issues/5523
7. https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf
8. https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
9. https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/
10. https://medium.com/diogo-menezes-borges/project-1-bigmart-sale-prediction-fdc04f07dc1e
11. https://rstudio-pubs-static.s3.amazonaws.com/381886_981132516a8e437284327a405ca4d91a.html
12. https://www.analyticsvidhya.com/blog/2016/02/bigmart-sales-solution-top-20/
13. https://medium.datadriveninvestor.com/k-nearest-neighbors-in-python-hyperparameters-tuning-716734bc557f
14. https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779
15. https://elitedatascience.com/data-cleaning
16. https://coderzcolumn.com/tutorials/data-science/missingno-visualize-missing-data-in-python
17. https://blog.usejournal.com/missing-data-its-types-and-statistical-methods-to-deal-with-it-5cf8b71a443f
18. https://towardsdatascience.com/exploratory-data-analysis-feature-engineering-and-modelling-using-supermarket-sales-data-part-1-228140f89298
19. https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
20. https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
21. https://medium.com/data-science-reporter/feature-selection-via-grid-search-in-supervised-models-4dc0c43d7ab1
22. https://stackoverflow.com/questions/45352420/avoid-certain-parameter-combinations-in-gridsearchcv
23. https://towardsdatascience.com/5-advanced-scikit-learn-features-that-will-transform-the-way-you-code-48262282ef0d
24. https://stackabuse.com/scikit-learn-save-and-restore-models/
25. https://medium.com/@jjosephmorrison/one-hot-encoding-to-set-up-categorical-features-for-linear-regression-6bac35661bb6#:~:text=One%2Dhot%20encoding%20is%20a,fit%20into%20the%20linear%20regression.&text=This%20is%20not%20something%20to,it%20can%20be%20an%20option.









