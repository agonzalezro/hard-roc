from numpy import arange
from data import draft, proof, eval, save, normalize

from sklearn.neighbors import KNeighborsClassifier

def do_knn(which = ''):
  (trX, trY, teX) = draft(which)
  (trX, teX) = normalize(trX, teX)

  clf = KNeighborsClassifier(probabilities = True)
  clf.fit(trX, trY)
  teY = clf.predict_proba(teX)[:,1]
  return teY
#save(teY)

print eval(do_knn(''))

save(teY)
