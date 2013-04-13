from numpy import loadtxt, mean, std, vstack
from sklearn.metrics import auc, roc_curve

def normalize(trX, teX):
  ms = mean(vstack((trX,teX)), 0)
  sd = std(vstack((trX,teX)), 0)
  trX = (trX - ms) / sd
  teX = (teX - ms) / sd
  return (trX, teX)

def proof(): 
  # load test/train data
  tr = loadtxt('train.csv', delimiter=',', skiprows=1)
  trY = tr[:, 0]
  trX = tr[:, 1:]
  teX = loadtxt('test.csv', delimiter=',', skiprows=1)

  (trX, teX) = normalize(trX, teX)
  return (trX, trY, teX)

def draft():
  # load test/train data
  tr = loadtxt('train.csv', delimiter=',', skiprows=1)
  trY = tr[:, 0]
  trX = tr[:, 1:]

  # first 4000 training we will use for local training
  ltrX = trX[:4000,:]
  ltrY = trY[:4000]

  # last 1500 training we will use as local hold-out
  lteX = trX[4000:,:]
  lteY = trY[4000:]

  (ltrX, lteX) = normalize(ltrX, lteX)

  return (ltrX, ltrY, lteX)

def eval(pteY):
  # load test/train data
  tr = loadtxt('train.csv', delimiter=',', skiprows=1)
  trY = tr[:, 0]
  lteY = trY[4000:]
  ACC = float(sum(lteY==(pteY>.5)))/pteY.shape[0]

  (fpr, tpr, thresholds) = roc_curve(lteY, pteY)
  AUC = auc(fpr, tpr)
  return (AUC, ACC)

