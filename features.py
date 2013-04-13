from data import draft, identify, eval, proof
from counts import adj
from numpy import vstack, array, log

identities, counts, total = adj([8])
(trX, trY, teX) = proof()

trf = open('new_train.csv', 'w')
print >>trf, ',lambda'

tef = open('new_test.csv', 'w')
print >>tef, ',lambda'

counts_test = []
counts_train = []
trY = []
for i in range(trX.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]

  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)
  features = []
  for jj in range(11):
    features.append(trX[i, jj] - trX[i, jj+11])
    features.append(log(1+trX[i, jj]) - log(1+trX[i, jj+11]))
    features.append(counts[le])
    features.append(counts[ri])

  p = ple / (ple + pri)
  features.append(p)

  print >>trf, ',' + ','.join(map(str, features))

teY = []
stacked = vstack((trX, teX))
for i in range(trX.shape[0], stacked.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]

  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)

  counts_test.append(counts[le])
  counts_test.append(counts[ri])
  p = ple / (ple + pri)

  features = []
  for jj in range(11):
    features.append(stacked[i, jj] - stacked[i, jj+11])
    features.append(log(1+stacked[i, jj]) - log(1+stacked[i, jj+11]))
    features.append(counts[le])
    features.append(counts[ri])

  p = ple / (ple + pri)
  features.append(p)
  print >>tef, ',' + ','.join(map(str, features))


teY = array(teY)
trf.close()
tef.close()
