
 ## Project Definition
 An overview of the project definition is described below. For more details see the competition 
 [Zillow Prize](https://www.kaggle.com/c/zillow-prize-1#description) 
 
 ### Project Overview
 <!-- 
    Student provides a high-level overview of the project in layman’s terms. Background information such as the 
    problem domain, the project origin, and related data sets or input data is given.
 -->
 I selected the competition [Zillow Prize](https://www.kaggle.com/c/zillow-prize-1#description) found at 
 [Kaggle](https://www.kaggle.com) as the Capstone project in my _Machine Learning Engineer Nanodegree_ provided 
 by [Udacity](https://www.udacity.com). 
 
 [Zillow](https://www.zillow.com) is the leading real estate and rental marketplace dedicated to empowering consumers 
 with data. They launched the Kaggle competition _Zillow Prize_ to improve their ability to predict house prices.
 
 I selected this competition to learn from the rich source of knowledge the community around these competitions 
 provide. Comparing and learning from public Kernels provided gives a good benchmark of cutting edge models 
 for these kind of problems.
 
 ### Problem Statement
 <!--
    The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion 
    of the expected solution, has been made.
 -->
 They have been developing their own model to predict prices for years. The problem at hand is to see if the 
 Kaggle community can improve on it creating an even better model.
 
 ### Metrics
 <!--
    Metrics used to measure performance of a model or result are clearly defined. Metrics are justified based on the 
    characteristics of the problem.
 -->
 The metric is defined by Zillow witch is asking to predict the _logerror_ between their Zestimate model and the 
 actual sale price, given all the features of a home. The log error is defined as
 ```
    logerror = log(Zestimate) − log(SalePrice)
 ```                                                
 
 
 ## Analysis
  
 ### Data Exploration
 <!-- 
    If a dataset is present, features and calculated statistics relevant to the problem have been reported and 
    discussed, along with a sampling of the data. In lieu of a dataset, a thorough description of the input space or 
    input data has been made. Abnormalities or characteristics about the data or input that need to be addressed 
    have been identified.
 -->
 This data exploration is inspired by the [Simple Exploration Notebook - Zillow Prize](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize)
 written by Kaggle Grandmaster [SRK](https://www.kaggle.com/sudalairajkumar).
 
 We will be working with following data files
 - _properties_2016.csv_ - all the properties with their home features for 2016. Note: Some 2017 new properties don't have any data yet except for their parcelid's. Those data points should be populated when properties_2017.csv is available.
 - _train_2016.csv_ - the training set with transactions from 1/1/2016 to 12/31/2016.
 - _sample_submission.csv_ - a sample submission file in the correct format


 #### Traning Set File
 The training set in file _train_2016.csv_ contains 90275 rows and three columns
 - _parcelid_ - which is the id of the property. 
 - _logerror_ - the log-error comparing the log of the actual price and the log of the predicted price.
 - _transactiondate_ - is the date when the property was sold
 Each row correspond to a property transaction and there will be more than one row in this file with the same 
 parcelid if the property has been sold more than ones during 2016. In fact counting there are 
 - 90026 properties that have been sold ones
 - 123 properties that have been sold twice
 - and one property that was sold one time.
 
 ##### Logerror
 So let us take a look at the distribution of the logerror
 ![](./train-logerror.png)
 A nice normal distribution centered around zero.
 
 ##### Transaction date
 According to Zillow the training data has all the transactions before October 15, 2016, plus some of the transactions 
 after October 15, 2016.
 ![](./train-transactiondate.png)
 From January to September there are about 6000-11000 transactions per month while dropping to under 2000 in 
 November december. 
 
 #### Properties
 Property data file _properties_2016.csv_ is a lat larger with 2985217 rows and 58 columns describing home features. 
 The file seems like some of the columns contain less information in the form of NaN values. Lets us look at that 
 more closely.
 ![](./prop-nan.png)
 Just looking at the data visually more than half of the data is not there and the data loss is unevenly distributed 
 on the featues. Even if we have a lot of features about 30 percent will most likely not contribute to a better 
 model when they do not contain any information. 
 
 
 ### Exploratory Visualization
 ```
 A visualization has been provided that summarizes or extracts a relevant characteristic or feature about the dataset
 or input data with thorough discussion. Visual cues are clearly defined.
 ```
 
 ### Algorithms and Techniques
 ```
 Algorithms and techniques used in the project are thoroughly discussed and properly justified based on the characteristics of the problem.
 ```
 
 ### Benchmark
 ```
 Student clearly defines a benchmark result or threshold for comparing performances of solutions obtained.
 ```
 
 ## Methodology
 
 ### Data Preprocessing
 ```
 All preprocessing steps have been clearly documented. Abnormalities or characteristics about the data or input that needed to be addressed have been corrected. If no data preprocessing is necessary, it has been clearly justified.
 ```
 
 ### Implementation
 ```
 The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.
 ```
 
 ### Refinement
 ```
 The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.
 ```
 
 ## Results
 
 ### Model Evaluation and Validation
 ```
 The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.
 ```
 ### Justification
 ```
 The final results are compared to the benchmark result or threshold with some type of statistical analysis. Justification is made as to whether the final model and solution is significant enough to have adequately solved the problem.
 ```
 
 ## Conclusion
 
 ### Free-Form Visualization
 ```
 A visualization has been provided that emphasizes an important quality about the project with thorough discussion. Visual cues are clearly defined.
 ```
 
 ### Reflection
 ```
 Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.
 ```
 
 ### Improvement
 ```
 Discussion is made as to how one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.
 ```
 