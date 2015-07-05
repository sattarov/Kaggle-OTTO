setwd("D:\\DOCS\\Kaggle\\OTTO\\raw_data")
library(caret)
library(foreach)
library(doParallel)
library(h2o)
training = read.csv("train.csv", header = T, na.string=c("","NA"));
testing = read.csv("test.csv", header = T, na.string=c("","NA"));

#Preprocessing
#Remove the first id column
training = training[,-1]
testing = testing[,-1]
nzv_features = nearZeroVar(training)
nzv_table = nearZeroVar(training, saveMetrics = T)

#Removing near zero variance features from the dataset
training = training[,-nzv_features]
testing = testing[,-nzv_features]

#scaling between 0-1
features = c(1:(ncol(training)-1))
training[,features] = apply(training[,features], 2, function(col) (col-mean(col))/(max(col)-min(col)) )

#Partition data on training and cross-validation sets
inTrain = createDataPartition(y=training$target, p=0.75, list=FALSE)
cv = training[-inTrain,]
training = training[inTrain,]


#add a constant and take sqrt
training[,features] = sqrt(training[,features] + (3/8))
cv[,features] = sqrt(cv[,features] + (3/8))

localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '6g',nthreads = 3)
training_h2o <- as.h2o(localH2O, training, key = 'train')
cv_h2o = as.h2o(localH2O, cv, key = 'cv')
init.time <- Sys.time()
model <- h2o.deeplearning(x = 1:(ncol(training_h2o)-1),  # column numbers for predictors
                          y = "target",   # column number for label
                          data = training_h2o, # data in H2O format
                          activation="TanhWithDropout",
                          classification = T,
                          input_dropout_ratio = 0.05, # % of inputs dropout
                          hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                          #balance_classes = TRUE, 
                          #variable_importance = T,
                          l1=1e-5,
                          l2=1e-5,
                          hidden = c(1024,512,256), # three layers of 50 nodes
                          epochs = 50) # max. no. of epochs
h2o_prediction <- as.data.frame(h2o.predict(model, cv_h2o))
confusionMatrix(h2o_prediction$predict, cv$target)
Sys.time() - init.time

#random forest
init.time <- Sys.time()
model = h2o.randomForest(x = 1:(ncol(training_h2o)-1),  # column numbers for predictors
                         y = "target",   # column number for label
                         data = training_h2o, # data in H2O format
                         classification = T,
                         ntree = 200,
                         depth = 50,
                         mtries = 25)
h2o_prediction <- as.data.frame(h2o.predict(model, cv_h2o))
h2o_prediction_prob = h2o_prediction[,2:ncol(h2o_prediction)]
confusionMatrix(h2o_prediction$predict, cv$target)
Sys.time() - init.time
actual_labels = true_Labels(h2o_prediction_prob, cv$target)
error = MultiLogLoss(actual_labels, h2o_prediction_prob)
error
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

nlevels(h2o_prediction$predict)
sum(h2o_prediction$predict==cv$target)
accuracies_cv = c(accuracies_cv, sum(h2o_prediction$predict==cv$target)/nrow(cv))
h2o_prediction <- as.data.frame(h2o.predict(model, training_h2o))
accuracies_train = c(accuracies_train, sum(h2o_prediction$predict==training$target)/nrow(training))

