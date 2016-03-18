---
title: "Practical Machine Learning"
author: "AmirYazid"
date: "Thursday, March 17, 2016"
---
##Data preprocessing
```r
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```

##Read Data
```{r}
trainRaw <- read.csv("D:/coursera/data/8 Practical Machine Learning/pml-training.csv")
testRaw <- read.csv("D:/coursera/data/8 Practical Machine Learning/pml-testing.csv")

dim(trainRaw)
dim(testRaw)
```

##Data Cleaning

###remove NA values
```{r}
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
```

###get rid of some columns that do not contribute much to the accelerometer measurements
```{r}
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```

###split the cleaned training set into a pure training data set (70%) and a validation data set (30%)
###use the validation data set to conduct cross validation in future steps
```{r}
set.seed(22519) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

##Data Modelling

###Fit a predictive model for activity recognition using Random Forest algorithm 
###because it automatically selects important variables and is robust to correlated covariates & outliers in general
###We will use 5-fold cross validation when applying the algorithm.

```{r}
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```

###Then estimate the performance of the model on the validation data set

```{r}
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
```

###calculate accuracy

```{r}
accuracy <- postResample(predictRf, testData$classe)
accuracy
```

###calculate estimated out-of-sample error

```{r}
oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose
```

##Predicting for Test Data Set

###apply the model to the original testing data set downloaded from the data source
###remove the problem_id column first.

```{r}
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```


##Appendix

###Correlation Matrix Visualization

```r
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```
	
	
###Tree Visualization

```r
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) # fast plot
```
