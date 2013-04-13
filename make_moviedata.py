from data import draft, identify, eval
from counts import adj
from numpy import vstack, array, log

identities, counts, total = adj([8])
(trX, trY, teX) = draft()

from data import lteY

train_vec = open('train_vec', 'w')

for i in range(trX.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]

  train_vec.write(str(le) + ' ' + str(ri))
  if trY[i] == 0:
    train_vec.write('1\n')

  else:
    train_vec.write('5\n')

train_vec.close()

probe_vec = open('probe_vec.txt', 'w')

teY = []
stacked = vstack((trX, teX))
for i in range(trX.shape[0], stacked.shape[0]):
  le = identities[i][0]
  ri = identities[i][1]

  probe_vec.write(str(le) + ' ' + str(ri))
  if lteY[i-trX.shape[0]] == 0:
    probe_vec.write('1\n')

  else:
    probe_vec.write('5\n')

probe_vec.close()
