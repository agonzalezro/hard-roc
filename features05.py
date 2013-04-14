from data import draft, identify, eval, proof
from counts import adj
from numpy import vstack, array, log

(trX, trY, teX) = proof()
identities, counts, total = adj(trX, trY, teX, [8])

trf = open('train_f06.csv', 'w')
print >>trf, 'header'

tef = open('test_f06.csv', 'w')
print >>tef, 'header'

for i in range(trX.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]
  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)
  p = ple / (ple + pri)

  features = [int(trY[i])]
  for jj in range(11):
    features.append(log(1+trX[i, jj]) - log(1+trX[i, jj+11]))

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
    features.append(log(1+stacked[i, jj]) - log(1+stacked[i, jj+11]))

  features.append(p)

  print >>tef, ','.join(map(str, features))

trf.close()
tef.close()
