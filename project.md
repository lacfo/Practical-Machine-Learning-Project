#Practical Machine Learning Final Project

##Objective

The object of this project is to predict how well the users perform barbells.

##Load the data


```r
setwd("d:/coursera/practical ML/project")
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
training <- read.csv("pml-training.csv", na.strings = c("", NA))
testing <- read.csv("pml-testing.csv", na.strings = c("", NA))
```

##Cleaning data

I remove the features in testing data set that are all NAs and the first 5 columns since they
are related to time and the last one which is problem id


```r
problemid <- testing$problem_id
features <- names(testing[, colSums(is.na(testing)) == 0])
##remove the first 7 and the last
features <- features[8 : 59]
testing <- testing[, features]
```

I select the training data features according to the testing set


```r
classes <- training$classe
training <- training[, features]
training <- cbind(classes, training)
```

##Do the classification

Split the training data into training and test set


```r
intrain <- createDataPartition(y = training$classes, p = 0.7, list = FALSE)
training <- training[intrain, ]
testing1 <- training[-intrain, ]
```

###First use the boosting


```r
set.seed(12345)
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)
modgbm <- train(classes~., data = training, method = "gbm",
                trControl = fitControl,
                verbose = FALSE)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
predgbm <- predict(modgbm, newdata = testing1)
confusionMatrix(predgbm, testing1$classes)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1146   24    0    0    0
##          B    7  768   19    2    5
##          C    0   23  695   14    3
##          D    0    1    4  683    8
##          E    0    0    0    4  709
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9723          
##                  95% CI : (0.9668, 0.9771)
##     No Information Rate : 0.2802          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.965           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9939   0.9412   0.9680   0.9716   0.9779
## Specificity            0.9919   0.9900   0.9882   0.9962   0.9988
## Pos Pred Value         0.9795   0.9588   0.9456   0.9813   0.9944
## Neg Pred Value         0.9976   0.9855   0.9932   0.9942   0.9953
## Prevalence             0.2802   0.1983   0.1745   0.1708   0.1762
## Detection Rate         0.2785   0.1866   0.1689   0.1660   0.1723
## Detection Prevalence   0.2843   0.1947   0.1786   0.1691   0.1733
## Balanced Accuracy      0.9929   0.9656   0.9781   0.9839   0.9884
```

###Then I use the classification tree


```r
set.seed(12345)
modtree <- train(classes~., data = training, method = "rpart")
```

```
## Loading required package: rpart
```

```r
predtree <- predict(modtree, newdata = testing1)
confusionMatrix(predtree, testing1$classes)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1061  352  361  334  110
##          B   17  257   18  127   85
##          C   71  207  339  242  197
##          D    0    0    0    0    0
##          E    4    0    0    0  333
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4836         
##                  95% CI : (0.4682, 0.499)
##     No Information Rate : 0.2802         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3254         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9202  0.31495  0.47214   0.0000  0.45931
## Specificity            0.6094  0.92513  0.78893   1.0000  0.99882
## Pos Pred Value         0.4784  0.50992  0.32102      NaN  0.98813
## Neg Pred Value         0.9515  0.84520  0.87610   0.8292  0.89624
## Prevalence             0.2802  0.19830  0.17448   0.1708  0.17618
## Detection Rate         0.2578  0.06245  0.08238   0.0000  0.08092
## Detection Prevalence   0.5390  0.12248  0.25662   0.0000  0.08190
## Balanced Accuracy      0.7648  0.62004  0.63054   0.5000  0.72907
```

###Then I use the linear discriminant analysis


```r
set.seed(12345)
modlda <- train(classes~., data = training, method = "lda")
```

```
## Loading required package: MASS
```

```r
predlda <- predict(modlda, newdata = testing1)
confusionMatrix(predlda, testing1$classes)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 946 133  72  28  30
##          B  27 536  55  25 120
##          C  94  83 494  87  73
##          D  81  24  79 534  65
##          E   5  40  18  29 437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7162          
##                  95% CI : (0.7021, 0.7299)
##     No Information Rate : 0.2802          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6409          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8205   0.6569   0.6880   0.7596   0.6028
## Specificity            0.9112   0.9312   0.9008   0.9270   0.9729
## Pos Pred Value         0.7825   0.7025   0.5945   0.6820   0.8261
## Neg Pred Value         0.9288   0.9165   0.9318   0.9493   0.9197
## Prevalence             0.2802   0.1983   0.1745   0.1708   0.1762
## Detection Rate         0.2299   0.1303   0.1200   0.1298   0.1062
## Detection Prevalence   0.2938   0.1854   0.2019   0.1903   0.1286
## Balanced Accuracy      0.8658   0.7940   0.7944   0.8433   0.7878
```

##Finally I predict the outcome in the testing data set


```r
predictions <- predict(modgbm, newdata = testing)
results <- data.frame(problemid, predictions)
results
```

```
##    problemid predictions
## 1          1           B
## 2          2           A
## 3          3           B
## 4          4           A
## 5          5           A
## 6          6           E
## 7          7           D
## 8          8           B
## 9          9           A
## 10        10           A
## 11        11           B
## 12        12           C
## 13        13           B
## 14        14           A
## 15        15           E
## 16        16           E
## 17        17           A
## 18        18           B
## 19        19           B
## 20        20           B
```

