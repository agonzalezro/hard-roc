from numpy import loadtxt, mean, std, vstack
from sklearn import svm
from sklearn.metrics import auc, roc_curve

# load test/train data
tr = loadtxt('train.csv', delimiter=',', skiprows=1)
trY = tr[:, 0]
trX = tr[:, 1:]
teX = loadtxt('test.csv', delimiter=',', skiprows=1)

# normalize
ms = mean(vstack((trX,teX)), 0)
sd = std(vstack((trX,teX)), 0)
trX = (trX - ms) / sd
teX = (teX - ms) / sd

# first 4000 training we will use for local training
ltrX = trX[:4000,:]
ltrY = trY[:4000]

# last 1500 training we will use as local hold-out
lteX = trX[4000:,:]
lteY = trY[4000:]

clf = svm.SVC(probability=True)
clf.fit(ltrX, ltrY)
lteY_svm01 = clf.predict_proba(lteX)[:,1]

(fpr, tpr, thresholds) = roc_curve(lteY, lteY_svm01)
print 'AUC:', auc(fpr, tpr)
print 'ACC:', float(sum(lteY==(lteY_svm01>.5)))/lteY_svm01.shape[0]
