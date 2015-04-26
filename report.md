# Practical Machine Learning Course Project:

## Synopsis
Using devices such as _Jawbone Up_, _Nike FuelBand_, and _Fitbit_ it is now possible to collect a large amount of data about personal activity relatively inexpensively. 
These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). 
The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

The goal of this project is to predict the manner in which the participants did the exercise.
In this report we present step-by-step the methods used and the results obtained.
We first detail the acquisition and basic processing of the data, then we estimate the out-of-sample performance for 2 different machine learning algorithms, namely Regression Tree and Random Forest.
As we observe that Random Forest seems to lead to better performance, we build our final model using this method.
We finally compute and submit the predicted classes for the test data.

## Loading and Processing the data

We first load all the libraries used for our code to run, and set the seed of R's random number generator for reproducibility purposes:

```r
library(caret)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
set.seed(35739)
```

We then load the data into R:

```r
dataTrain <- read.csv("pml-training.csv", header=T, na.strings = c("NA", "#DIV/0!" ,""))
dataTest  <- read.csv("pml-testing.csv", header=T, na.strings = c("NA", "#DIV/0!" ,""))
```

The 2 datasets consit of respectively 19622 and 20 observations of 160 variables:

```r
colnames(dataTrain)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

We first discard the variables that are of no interest for our classification problem (_eg._ observation number, user name, time stamps ...):

```r
dataTrain <- dataTrain[,-c(1:7)]
dataTest  <- dataTest[,-c(1:7)]
```

We then discard the variables for which there is 50% or more missing values in the training set:

```r
pcNA      <- apply(dataTrain, 2, function(x) sum(is.na(x))/length(x) )
dataTrain <- dataTrain[, pcNA<0.5]
dataTest  <- dataTest[, pcNA<0.5]
```

We are left with 53 variables.
We then remove near zeo variance variables (from the training set):

```r
nzv       <- nearZeroVar(dataTrain, saveMetrics=TRUE)
dataTrain <- dataTrain[,!nzv$nzv]
dataTest  <- dataTest[,!nzv$nzv]
```
We are still left with 53 variables (which means that no additional variable was removed, it is still good practice to anyway be checking for it).

One last adjustment is to discard the last column of the test dataset which is the problem ID (in the training dataset this last column corresponds to the class of the observation):

```r
dataTest  <- dataTest[,-ncol(dataTest)]
```

## Machine Learning Algorithm Selection

We aim here at selecting the best machine learning algorithm for our classification problem.
We want to use the method that will lead to the smallest out-ot-sample error.
To this effect split the original training set into a 70% training set and 30% testing set.
The new training set is used to train various machine learning algorithms, and the test set is used to measure the classification performance and in this way to estimate the out-of-sample error.
We consider 2 machine learning algorithms: a regression tree and a random forest classifier.
For both we perform the training procedure via a 5-fold cross-validation:

```r
inTrain     <- createDataPartition(y=dataTrain$classe, p=0.7, list=FALSE)
myTraining  <- dataTrain[inTrain, ]
myTesting   <- dataTrain[-inTrain, ]
trCtrl      <- trainControl(method = "cv", number=5)
```

### Regression tree
We show here the results from the regression tree:

```r
mTree <- train( classe ~ ., data=myTraining, method="rpart", trControl = trCtrl)
```

```
## Loading required namespace: e1071
```

```r
fancyRpartPlot(mTree$finalModel)
```

![](report_files/figure-html/unnamed-chunk-9-1.png) 

```r
predictionsTree <- predict(mTree, newdata=myTesting)
confusionMatrix(predictionsTree, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1529  470  476  430  163
##          B   25  396   32  161  148
##          C  114  273  518  373  284
##          D    0    0    0    0    0
##          E    6    0    0    0  487
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4979         
##                  95% CI : (0.485, 0.5107)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3436         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9134  0.34767  0.50487   0.0000  0.45009
## Specificity            0.6345  0.92288  0.78514   1.0000  0.99875
## Pos Pred Value         0.4984  0.51969  0.33163      NaN  0.98783
## Neg Pred Value         0.9485  0.85497  0.88249   0.8362  0.88965
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2598  0.06729  0.08802   0.0000  0.08275
## Detection Prevalence   0.5213  0.12948  0.26542   0.0000  0.08377
## Balanced Accuracy      0.7740  0.63528  0.64501   0.5000  0.72442
```

This first classification attempt shows quite poor results; the accuracy barely reaches 50% accuracy, all observations from class D are misclassified and only data from class A have a sensitivity higher than 50%.
We can therefore only hope that our next attempt to build a classifier will show better performance o the test.

### Random forest
We show here the results from the random forest:

```r
mRF <- train( classe ~ ., data=myTraining, method="rf", trControl = trCtrl)
predictionsRF <- predict(mRF, newdata=myTesting)
confusionMatrix(predictionsRF, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    7    0    0    0
##          B    0 1131    5    1    0
##          C    0    1 1015   20    2
##          D    0    0    6  943    6
##          E    0    0    0    0 1074
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9918         
##                  95% CI : (0.9892, 0.994)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9897         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9930   0.9893   0.9782   0.9926
## Specificity            0.9983   0.9987   0.9953   0.9976   1.0000
## Pos Pred Value         0.9958   0.9947   0.9778   0.9874   1.0000
## Neg Pred Value         1.0000   0.9983   0.9977   0.9957   0.9983
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1922   0.1725   0.1602   0.1825
## Detection Prevalence   0.2856   0.1932   0.1764   0.1623   0.1825
## Balanced Accuracy      0.9992   0.9959   0.9923   0.9879   0.9963
```

The random forest showq very good results: an accuracy of about 99% and sentitivities and specificities of at least 99% for each of our 5 classes.


## Building the final model and submitting the results

As the random forest method led to lower values for the out-of-sample error estimate, we select this classification method.
We thus build a random forest classifier on the full training dataset this time, and apply it to our original test set.
We show the predictions of the model on the test set:

```r
mRF_final <- train( classe ~ ., data=dataTrain, method="rf", trControl = trCtrl)
predictionsRF <- predict(mRF, newdata=dataTest)
predictionsRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The following code generate files for the assignment submission:

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```

