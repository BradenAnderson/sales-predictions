
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

###### [Data Cleaning](#data-cleaning)

The code for this section can be found in the file named 01_Sales_Predictions_Data_Cleaning.ipynb

This notebook takes as an input a csv file that contains the BigMart sales dataset. The csv file is read into a pandas DataFrame, which is then used to explore and gain familiarity with the various features and observations the dataset contains. Early exploration steps involve viewing the data types of each feature, checking for duplicated records, and reviewing the various categories within each feature to see if observations were consistently recorded. As a quick aside, this project exclusively used the built in Pandas library functions for the early data exploration steps, however much of this can be easily automated using the Pandas Profiling library, which if interested can be found [here](https://github.com/pandas-profiling/pandas-profiling). Next, the records that contain missing values were viewed in two different ways, first by performing Pandas DataFrame filtering and second by creating visualizations using the python library missingno. This review of missing values showed that 2 of the 11 features contained missing information, 'Outlet Size' and 'Item Weight'. Additionally, we gained some insight by noticing that these missing values did not appear to be entirely random, as the majority of the missing weight values came from the earliest year in the dataset, 1985 (maybe they didn't have scales back then?). This could be indicative of an error in the data collection or aggregation process, therefore in a real scenario this is the point where the data scientist should pause and reach out to the subject matter expert or individual who supplied the data to see if there is a way to obtain the correct information. In this situation I assumed the correct values could not be obtained and proceeded to fill in the missing information with the help of Scikit-Learns Simpleimputer. The missing weight values were filled in using the average item weight value, and the missing outlet size values were flagged by filling in a value of "missing". 


***

###### [Exploratory Data Analysis](#exploratory-data-analysis)

The code for this section can be found in the file named 02_Sales_Predictions_EDA.ipynb

***

###### [Modeling Part 1](#modeling-part-1)
The code for this section can be found in the file named 03_Modeling_Part_1.ipynb

***

###### [Feature Engineering and Modeling Part 2](#feature-engineering-and-modeling-part-2) 
The code for this section can be found in the file named 04_Feature_Engineering_and_Modeling_Pt2.ipynb

***

###### [Interpretable Models](#interpretable-models) 
The code for this section can be found in the file named 05_Interpretable_Modeling.ipynb

***

###### [Tuning Hyperparameters](#tuning-hyperparameters) 
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









