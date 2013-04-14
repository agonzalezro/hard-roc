from data import save, eval, draft, proof
from counts import dead_simple

import itertools
def findsubsets(S,m):
    return set(itertools.combinations(S, m))

subsets = []
aucs = []
(trX, trY, teX) = draft()
stop = 20
at = 0
for subset in findsubsets(range(11), 4):
  subsets.append(subset)
  auc = eval(dead_simple(trX, trY, teX, subset))[0]
  aucs.append(auc)
  print subset, eval(dead_simple(trX, trY, teX, subset))
  at += 1 
  if at == stop:
    break

from numpy import array, sum
aucs = array(aucs)
aucs = aucs / sum(aucs)

(trX, trY, teX) = proof()
prediction = None

for (subset, auc) in zip(subsets, aucs):
  print auc
  if prediction == None:
    prediction = dead_simple(trX, trY, teX, subset)*auc

  else:
    prediction += dead_simple(trX, trY, teX, subset)*auc
    
save(prediction, 'orders/first5.csv')
