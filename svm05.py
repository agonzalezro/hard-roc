from sklearn import svm
from numpy import arange
from data import draft, proof, eval, save, normalize

def dosvm(gamma = 1.0):
  which = '_f05'
  (trX, trY, teX) = draft(which)
  (trX, teX) = normalize(trX, teX)
  clf = svm.SVC(probability = True, kernel='rbf', gamma = gamma, tol=0.00001)
  clf.fit(trX, trY)
  teY = clf.predict_proba(teX)[:,1]
  print eval(teY)

#for gamma in [0.001, 0.002, 0.004, 0.006, 0.008]:
#  dosvm(gamma)

#
(trX, trY, teX) = proof('_f05')
(trX, teX) = normalize(trX, teX)
clf = svm.SVC(probability=True, kernel='rbf', gamma=0.002, tol=0.00001)
clf.fit(trX, trY)
teY = clf.predict_proba(teX)[:,1]
save(teY, 'pred_svm06.csv')
