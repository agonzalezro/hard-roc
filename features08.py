from data import draft, identify, eval, proof
from counts import adj
from numpy import vstack, array, log

(trX, trY, teX) = proof()
identities, counts, total = adj(trX, trY, teX, [8])

trf = open('train_f08.csv', 'w')
print >>trf, 'header'

tef = open('test_f08.csv', 'w')
print >>tef, 'header'

def make_features(identity, x):
  features = []
  le = identity[0]
  ri = identity[1]
  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)
  p = ple / (ple + pri)
  features.append(p)
  features.append(counts[le])
  features.append(counts[ri])
  features.append(ple)
  features.append(pri)
  features.append(total[le])
  features.append(total[ri])

  for jj in range(11):
    #features.append(log(1+x[jj]) - log(1+x[jj+11]))
    #features.append(log(1+ (x[jj]+1) / (x[jj+11]+1)))
    features.append(x[jj]-x[jj+11])
    #features.append((x[jj]+1)/(x[jj+11]+1))

  for pair in [[0,1], [4,6], [3,5]]:
    l = pair[0]
    r = pair[1]
    features.append((1 + x[l]) / (x[l] + x[r] + 1))
    features.append((1 + x[1+11]) / (x[l+11] + x[r+11] + 1))


  return features

for i in range(trX.shape[0]):
  features = [int(trY[i])] + make_features(identities[i], trX[i,:])
  print >>trf, ','.join(map(str, features))

stacked = vstack((trX, teX))
for i in range(trX.shape[0], stacked.shape[0]):
  features = [] + make_features(identities[i], stacked[i, :])
  print >>tef, ','.join(map(str, features))

trf.close()
tef.close()
