from numpy import loadtxt, mean, std, vstack
from sklearn.metrics import auc, roc_curve

def identify(XX, which):
  to = dict()
  id = 1
  identities = []
  def make_key(v, which):
    key = []
    for w in which:
      key.append(v[w])

    return tuple(key)

  for i in range(XX.shape[0]):
    le = make_key(XX[i, :11], which)
    if not(le in to.keys()):
      to[le] = id
      id += 1

    ri = make_key(XX[i, 11:], which)
    if not(ri in to.keys()):
      to[ri] = id
      id += 1

    identities.append([to[le], to[ri]])

  return identities


def save(pred, filename = 'pred.csv'):
  # load test/train data
  teX = loadtxt('test.csv', delimiter=',', skiprows=1)
  pf = open(filename, 'w')
  for i in range(pred.shape[0]):
    print >>pf, ','.join([str(pred[i])] + map(str, list(teX[i, :])))

  pf.close()

def normalize(trX, teX):
  ms = mean(vstack((trX,teX)), 0)
  sd = std(vstack((trX,teX)), 0)
  trX = (trX - ms) / sd
  teX = (teX - ms) / sd
  return (trX, teX)

def proof(which = ''): 
  # load test/train data
  tr = loadtxt('train' + which + '.csv', delimiter=',', skiprows=1)
  trY = tr[:, 0]
  trX = tr[:, 1:]
  teX = loadtxt('test' + which + '.csv', delimiter=',', skiprows=1)
  return (trX, trY, teX)

from numpy.random import shuffle

lteY = None
def draft(which = ''):
  # load test/train data
  tr = loadtxt('train' + which + '.csv', delimiter=',', skiprows=1)
  if not('ix' in globals()):
    print 'shufflin'
    global ix
    ix = range(tr.shape[0])

  else:
    global ix

  tr = tr[ix, :]
  trY = tr[:, 0]
  trX = tr[:, 1:]

  # first 2000 training we will use for local training
  ltrX = trX[:2000,:]
  ltrY = trY[:2000]
  # last 1500 training we will use as local hold-out
  lteX = trX[2000:,:]
  global lteY
  lteY = trY[2000:]

  return (ltrX, ltrY, lteX)

def eval(pteY):
  # load test/train data
  #tr = loadtxt('train_f01.csv', delimiter=',', skiprows=1)
  #lteY = trY[4000:]
  global lteY

  (fpr, tpr, thresholds) = roc_curve(lteY, pteY)
  AUC = auc(fpr, tpr)
  ACC = float(sum(lteY==(pteY>.5)))/pteY.shape[0]
  return (AUC, ACC)

