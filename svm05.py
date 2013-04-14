from sklearn import svm
from numpy import arange
from data import draft, proof, eval, save, normalize

def do_draft(model, which):
  (trX, trY, teX) = draft(which)
  (trX, teX) = normalize(trX, teX)
  model.fit(trX, trY)                
  teY = model.predict_proba(teX)[:,1]
  print eval(teY)
  return model

def do_proof(model, which):
  (trX, trY, teX) = proof(which)
  (trX, teX) = normalize(trX, teX)
  model.fit(trX, trY)
  teY = model.predict_proba(teX)[:,1]
  save(teY, 'pred.csv')
  return model

which = '_ad01'
clf = svm.SVC(probability = True, kernel='rbf', gamma = 0.002, tol=0.00001)
do_draft(clf, which)

which = '_f08'
clf = svm.SVC(probability = True, kernel='rbf', gamma = 0.002, tol=0.00001)
do_draft(clf, which)

which = '_f09'
clf = svm.SVC(probability = True, kernel='rbf', gamma = 0.002, tol=0.00001)
do_draft(clf, which)
