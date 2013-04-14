from sklearn import svm
from numpy import arange
from data import draft, proof, eval, save, normalize

def do_svm(which = ''):
  (trX, trY, teX) = draft(which)
  (trX, teX) = normalize(trX, teX)

  clf = svm.NuSVC(probability=True, kernel='poly', degree=7, gamma=0.1)
  clf.fit(trX, trY)
  teY = clf.predict_proba(teX)[:,1]
  return teY

print eval(do_svm('_f03'))
