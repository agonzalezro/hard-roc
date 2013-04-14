library(randomForest)
library(foreach)
library(doMC)

## load data
test.set <- read.csv('/Users/adityajitta/Desktop/newTest.csv', header=T)
train.set <- read.csv('/Users/adityajitta/Desktop/newTrain.csv', header=T)

# Extract feature labels 
features <- names(train.set) 

## Split data into validation and test set

train.set_validation.index <- sample(nrow(train.set), nrow(train.set)/5, replace = F)
train.set_train <- train.set[-train.set_validation.index, features]
train.set_validation <- train.set[train.set_validation.index, features]

# Train model by combining 10 random Forests with 100 trees each.

rf.model<-foreach(ntree=rep(150,8), .combine=combine, .packages="randomForest", .inorder=F) %dopar% {randomForest(Choice~., data=train.set[,features], ntree=ntree)}

rf.model<-foreach(ntree=rep(1500,6), .combine=combine, .packages="randomForest", .inorder=F) %dopar% {randomForest(Choice~., data=train.set_train[,features], ntree=ntree,mtry=4)}

# Prediction probabilities on validation set found to be 0.98

k<-predict(rf.model, newdata=train.set_validation, type='response')

# get the fraction of correct prediction on validation set with a threshold of 0.5
y1map=k>0.5
ymap=train.set_validation$Choice==1
print (sum(ymap==y1map)/(sum(ymap!=y1map)+sum(ymap==y1map)))

# Prediction on test set

test.prediction <- predict(rf.model, newdata=test.set, type='response')

# writes the results onto a file, which is preprocessed using processSubmit.py
# to get the final submission file.
test.set <- transform(test.set, Choice=test.prediction)
write.csv(test.set, '/Users/adityajitta/Desktop/newFeatureSub1.csv', row.names=F)


# adaboost

gdis<-ada(Choice~.,train.set_train,iter=100,nu=0.5,type="discrete")
predict(gdis, newdata = train.set_validation, type = c("probs","F"),n.iter=NULL)
boost.pred<-predict(gdis, newdata = test.set, type = c("probs"),n.iter=NULL)
test.prediction<-boost.pred[,2]
test.set <- transform(test.set, Choice=test.prediction)
write.csv(test.set, '/Users/adityajitta/Desktop/subBoost1.csv', row.names=F)










