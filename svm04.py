from sklearn import svm
from numpy import arange
from data import draft, proof, eval, save, normalize

#(trX, trY, teX) = proof()
#clf = svm.SVC(probability=True, kernel='rbf', C = 5.0, gamma = 0.1)
#clf.fit(trX, trY)
#teY_svm01 = clf.predict_proba(teX)[:,1]
#save(teY_svm01)

def do_svm(which = ''):
  (trX, trY, teX) = draft(which)
  (trX, teX) = normalize(trX, teX)

  clf = svm.NuSVC(probability=True)
  clf.fit(trX, trY)
  teY = clf.predict_proba(teX)[:,1]
  return teY

def do_nusvm(nu, which = ''):
  (trX, trY, teX) = draft(which)
  (trX, teX) = normalize(trX, teX)

  clf = svm.NuSVC(probability=True)
  clf.fit(trX, trY)
  teY = clf.predict_proba(teX)[:,1]
  return teY



(trX, trY, teX) = proof('_f03')
(trX, teX) = normalize(trX, teX)

clf = svm.NuSVC(probability=True, nu=0.5)
clf.fit(trX, trY)
teY = clf.predict_proba(teX)[:,1]
save(teY)
