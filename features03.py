from data import draft, identify, eval, proof
from counts import adj
from numpy import vstack, array, log

(trX, trY, teX) = proof()
identities, counts, total = adj(trX, trY, teX, [8])

trf = open('train_f05.csv', 'w')
print >>trf, 'header'

tef = open('test_f05.csv', 'w')
print >>tef, 'header'

for i in range(trX.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]
  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)
  assert trY.ndim == 1
  features = [int(trY[i])]
  for jj in range(11):
    features.append(trX[i, jj] - trX[i, jj+11])
    features.append(log(1+trX[i, jj]) - log(1+trX[i, jj+11]))

  features.append(trX[i, 0] - trX[i, 1])
  features.append(trX[i, 11] - trX[i, 12])

  features.append(trX[i, 4] / (trX[i, 4] + trX[i, 6]))
  features.append(trX[i, 15] / (trX[i, 15] + trX[i, 17]))

  features.append(trX[i, 3] / (trX[i, 3] + trX[i, 5]))
  features.append(trX[i, 14] / (trX[i, 14] + trX[i, 16]))

  features.append(counts[le])
  features.append(counts[ri])
  features.append(ple)
  features.append(pri)

  p = ple / (ple + pri)
  features.append(p)

  print >>trf, ','.join(map(str, features))

stacked = vstack((trX, teX))
for i in range(trX.shape[0], stacked.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]

  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)

  p = ple / (ple + pri)
  features = []
  for jj in range(11):
    features.append(stacked[i, jj] - stacked[i, jj+11])
    features.append(log(1+stacked[i, jj]) - log(1+stacked[i, jj+11]))

  features.append(stacked[i, 0] - stacked[i, 1])
  features.append(stacked[i, 11] - stacked[i, 12])

  features.append(stacked[i, 4] / (stacked[i, 4] + stacked[i, 6]))
  features.append(stacked[i, 15] / (stacked[i, 15] + stacked[i, 17]))

  features.append(stacked[i, 3] / (stacked[i, 3] + stacked[i, 5]))
  features.append(stacked[i, 14] / (stacked[i, 14] + stacked[i, 16]))

  features.append(counts[le])
  features.append(counts[ri])
  features.append(ple)
  features.append(pri)

  p = ple / (ple + pri)
  features.append(p)
  print >>tef, ','.join(map(str, features))

trf.close()
tef.close()
