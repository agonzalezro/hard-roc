from data import draft, identify, eval, proof
from counts import adj
from numpy import vstack, array, log

(trX, trY, teX) = proof()
identities, counts, total = adj(trX, trY, teX, [8])

trf = open('train_f07.csv', 'w')
print >>trf, 'header'

tef = open('test_f07.csv', 'w')
print >>tef, 'header'

def make_features(identity, x):
  le = identity[0]
  ri = identity[1]
  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)
  p = ple / (ple + pri)

  features = []
  for jj in range(11):
    features.append(log(1+x[jj]) - log(1+x[jj+11]))
    features.append(log(1+ (x[jj]+1) / (x[jj+11]+1)))
    features.append(x[jj]-x[jj+11])
    features.append((x[jj]+1)/(x[jj+11]+1))

  for j1 in range(11):
    for j2 in range(11):
      features.append((x[j1]+1)*(x[j2]+1))
      features.append((x[j1+11]+1)*(x[j2+11]+1))

      features.append((x[j1+11]+1)*(x[j2]+1))
      features.append((x[j1]+1)*(x[j2+11]+1))

  features.append(log(1 + x[0] / (x[0] + x[1])))
  features.append(log(1 + x[11] / (x[11] + x[12])))

  features.append(log(1 + x[4] / (x[4] + x[6])))
  features.append(log(1 + x[15] / (x[15] + x[17])))
  #features.append(log(1 + x[7] / x[18]))

  features.append(p)
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
