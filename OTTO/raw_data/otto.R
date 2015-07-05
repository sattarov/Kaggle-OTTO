setwd("D:\\DOCS\\Kaggle\\OTTO\\raw_data")
library(kernlab)
library(caret)
library(foreach)
library(doParallel)
training = read.csv("train.csv", header = T, na.string=c("","NA"));
testing = read.csv("test.csv", header = T, na.string=c("","NA"));

#Preprocessing
#Remove the first id column
training = training[,-1]
testing = testing[,-1]
nzv_features = nearZeroVar(training)
nzv_table = nearZeroVar(training, saveMetrics = T)
#Plotting histograms of all features
setwd("../")
dir.create("feature_histograms")
setwd("feature_histograms")
getwd()
for (feature in colnames(training[,1:93]))
{
  jpeg(paste0(feature, ".jpeg"))
  hist(training[,feature])
  dev.off() 
}
#Plotting histograms of near zero variance features
dir.create("near_zero_var_feature_histograms")
setwd("near_zero_var_feature_histograms")
getwd()
for (feature in colnames(training[,nzv_features]))
{
  jpeg(paste0(feature, ".jpeg"))
  hist(training[,feature])
  dev.off() 
}

#Removing near zero variance features from the dataset
training = training[,-nzv_features]
testing = testing[,-nzv_features]
#Standardizing features
pre_Object = preProcess(training[,-81], method = "center", "scale")
stand = predict(pre_Object, training[,-81])
setwd("../")
dir.create("standardized_features")
setwd("standardized_features")
for (feature in colnames(training))
{
  jpeg(paste0(feature, ".jpeg"))
  hist(training[,feature])
  dev.off() 
}
log(0.1)
hist(stand[,5])
hist(log(stand[,5]))

nzv_table
mean(training[,-81])
a = (apply(training[,-81], 2, mean))
plot(apply(training[,-81], 2, sd))
plot(apply(stand, 2, sd))
warnings()

#Partition data on training and cross-validation sets
inTrain = createDataPartition(y=training$target, p=0.75, list=FALSE)
cv = training[-inTrain,]
training = training[inTrain,]

#summaryFunction for train function in caret
MultiLogLossCaret <- function(data, lev=NULL, model=NULL)
{
  eps = 1e-15;
  nr <- nrow(data)
  print(dim(data))
  pred = data[,3:11]
  #true_labels function
  #act = true_labels(pred, data[,"obs"])
  rows = dim(pred)[1]
  cols = dim(pred)[2]
  classes = names(pred)
  act_sample = matrix(rep(classes, rows), nrow = rows, byrow = T)
  act_labels = matrix(rep(data[,"obs"], cols), ncol = cols)
  act = (act_labels==act_sample)*1 
  
  pred = matrix(sapply( as.matrix(pred), function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( as.matrix(pred), function(x) min(1-eps,x)), nrow = nr)
  out = sum(act*log(pred) )
  out = out * -1/(nrow(act))      
  names(out)="MLL"
  return(out);
}

#Check correlated features
M = abs(cor(training[,-81]))
diag(M) = 0
which(M>0.8, arr.ind = TRUE)

#Reducing the dimentionality to 2D and plotting the dataset
prComp = prcomp(training[,-81], scale=TRUE)
plot(prComp$x[,1], prComp$x[,2], col=training$target, type="p", cex=0.5)

preProc = preProcess(training[,-81], method = "pca",thresh = 0.9)
training_preProc_PCA = predict(preProc, training[,-81])
plot(training_preProc_PCA[,1], training_preProc_PCA[,2], col=training$target)


modelInfo = getModelInfo(model="C5.0")
modelInfo$C5.0$grid
s=sigest(as.matrix(training[,-81]))
warnings()

#cl <- makeCluster(detectCores())
cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "repeatedcv", number = 3, repeats = 1, verboseIter = T, classProbs = T, summaryFunction=MultiLogLossCaret, allowParallel = T)
grid = expand.grid(mtry=41)
modelFit = train(target ~., data=training, method="C5.0",preProcess=c("scale","center"), trControl = trainControl, metric="MLL", maximize=F)
stopCluster(cl)

#try C5.0
modelInfo = getModelInfo(model="C5.0")
modelInfo$C5.0$grid

cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "none", 
                            number = 3, 
                            repeats = 1, 
                            verboseIter = T, 
                            classProbs = T, 
                            allowParallel = T)
grid = expand.grid(model="rule", trials=20, winnow=F)
modelFit = train(target ~., 
                 data=training, 
                 method="C5.0",
                 tuneGrid = grid, 
                 trControl = trainControl, 
                 metric="Accuracy")
stopCluster(cl)

#try CART
modelInfo = getModelInfo(model="rpart")
modelInfo$rpart$grid

cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "none", 
                            number = 3, 
                            repeats = 1, 
                            verboseIter = T, 
                            classProbs = T, 
                            allowParallel = T)
grid = expand.grid(cp=0.000005)
modelFit = train(target ~., 
                 data=training, 
                 method="rpart",
                 tuneGrid = grid, 
                 trControl = trainControl, 
                 metric="Accuracy")
stopCluster(cl)

predictions = predict(modelFit, newdata = cv)
predictions_prob = predict(modelFit, newdata = cv, "prob")
confusionMatrix(predictions, cv$target)
save(modelFit, file="rf_repeatedcv")
actual_labels = true_Labels(predictions_prob, cv$target)
error = MultiLogLoss(actual_labels, predictions_prob)

#Generate a table of actual labels
true_Labels = function(probs, labels)
{
  rows = dim(probs)[1]
  cols = dim(probs)[2]
  classes = names(probs)
  act_sample = matrix(rep(classes, rows), nrow = rows, byrow = T)
  act_labels = matrix(rep(labels, cols), ncol = cols)
  act = (act_labels==act_sample)*1
  return (act);
}


#Apply the model on a test set
predictions_testing = predict(modelFit, newdata = testing)
predictions_prob_testing = predict(modelFit, newdata = testing, "prob")
write.csv(predictions_prob_testing, file="submission4.csv", quote=F)
head(predictions_prob_testing)



trainControl = trainControl(method = "repeatedcv", number = 3, repeats = 1, verboseIter = T, classProbs = T, summaryFunction=MultiLogLossCaret)
grid = expand.grid(mtry=41)
#try PCA with 90% of variance
preProc = preProcess(training[,-81], method = "pca",thresh = 0.9)
target_training = training$target
training = predict(preProc, training[,-81])
training$target = target_training
target_cv = cv$target
cv = predict(preProc, cv[,-81])
cv$target = target_cv

modelFit = train(target ~., data=training, method="rf",preProcess=c("scale","center"), trControl = trainControl, tuneGrid = grid, metric="MLL", maximize=F)
predictions = predict(modelFit, newdata = cv)
predictions_prob = predict(modelFit, newdata = cv, "prob")
confusionMatrix(predictions, cv$target)
save(modelFit, file="rf_pca_repeatedcv")
actual_labels = true_Labels(predictions_prob, cv$target)
error = MultiLogLoss(actual_labels, predictions_prob)


#Multilog loss function
MultiLogLoss <- function(act, pred)
{
  eps = 1e-15;
  nr <- nrow(pred)
  pred = matrix(sapply( as.matrix(pred), function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( as.matrix(pred), function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) )
  ll = ll * -1/(nrow(act))      
  return(ll);
}


