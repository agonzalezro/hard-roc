from data import draft, identify, eval
from numpy import unique, array, vstack

(trX, trY, teX) = draft()
identities = identify(vstack((trX, teX)), [8])
counts = dict()
total = dict()

for id in list(unique(array(identities))):
  counts[id] = 0.0
  total[id] = 0.0

for i in range(trX.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]

  if trY[i] == 1:
    counts[le] += 1.0

  else:
    counts[ri] += 1.0

  total[le] += 1.0
  total[ri] += 1.0

teY = []

for i in range(trX.shape[0],vstack((trX, teX)).shape[0]):
  le = identities[i][0]
  ri = identities[i][1]

  ple = (counts[le]+1) / (total[le]+1)
  pri = (counts[ri]+1) / (total[ri]+1)

  teY.append(ple / (ple + pri))

teY = array(teY)

print eval(teY)
