from sklearn import svm
from numpy import arange
from data import draft, proof, eval, save

#(trX, trY, teX) = proof()
#clf = svm.SVC(probability=True, kernel='rbf', C = 5.0, gamma = 0.1)
#clf.fit(trX, trY)
#teY_svm01 = clf.predict_proba(teX)[:,1]
#save(teY_svm01)


(trX, trY, teX) = draft()
print 'C gamma degree AUC ACC'
for myC in [0.1, 1.0, 5.0, 10.0]:
  for mygamma in [0.01, 0.1, 1.0, 5.0, 10.0]:
    for mydegree in [2, 3, 4, 5]:
      clf = svm.SVC(probability=True, kernel='poly', C = myC, gamma = mygamma, degree = mydegree)
      clf.fit(trX, trY)
      teY_svm01 = clf.predict_proba(teX)[:,1]
      print myC, mygamma, mydegree, eval(teY_svm01)
