from data import draft, identify, eval, proof
from numpy import unique, array, vstack

def dead_simple(trX, trY, teX, w):
  identities, counts, total = adj(trX, trY, teX, w)
  teY = []
  for i in range(trX.shape[0],vstack((trX, teX)).shape[0]):
    le = identities[i][0]
    ri = identities[i][1]

    ple = (counts[le]+1) / (total[le]+1)
    pri = (counts[ri]+1) / (total[ri]+1)

    teY.append(ple / (ple + pri))

  teY = array(teY)
  return teY



def adj(trX, trY, teX, w):
  identities = identify(vstack((trX, teX)), w)
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

  return identities, counts, total

#for i in range(11):
#  which([i])
