library(randomForest)
library(foreach)
library(doMC)

## load data
tSet <- read.csv('/Users/adityajitta/Desktop/test.csv', header=T)
trSet <- read.csv('/Users/adityajitta/Desktop/train.csv', header=T)

# Extract feature labels 
features <- names(trSet) 

## Split data into validation and test set

trSet_validation.index <- sample(nrow(trSet), nrow(trSet)/5, replace = F)
trSet_train <- trSet[-trSet_validation.index, features]
trSet_validation <- trSet[trSet_validation.index, features]

# Train model by combining 10 random Forests with 100 trees each.

rf.model<-foreach(ntree=rep(100,10), .combine=combine, .packages="randomForest", .inorder=F) %dopar% {randomForest(Choice~., data=trSet_train[,features], ntree=ntree, importance=T, na.action=na.roughfix)}

# Prediction probabilities on validation set found to be 0.98

k<-predict(rf.model, newdata=trSet_validation, type='response')

# get the fraction of correct prediction on validation set with a threshold of 0.5
y1map=k>0.5
ymap=trSet_validation$Choice==1
print (sum(ymap==y1map)/(sum(ymap!=y1map)+sum(ymap==y1map)))

# Prediction on test set
test.prediction <- predict(rf.model, newdata=tSet, type='response')

# writes the results onto a file, which is preprocessed using processSubmit.py
# to get the final submission file.
write.csv(test.prediction, '/Users/adityajitta/Desktop/subRF.csv', row.names=F)