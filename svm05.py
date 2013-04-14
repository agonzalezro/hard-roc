from sklearn import svm
from numpy import arange
from data import draft, proof, eval, save, normalize

which = '_f05'
(trX, trY, teX) = draft(which)
#(trX, teX) = normalize(trX, teX)

clf = svm.NuSVC(probability=True, kernel='linear')
clf.fit(trX, trY)
teY = clf.predict_proba(teX)[:,1]
print eval(teY)

#
#(trX, trY, teX) = proof('_f04')
#(trX, teX) = normalize(trX, teX)
#clf = svm.NuSVC(probability=True, kernel='linear')
#clf.fit(trX, trY)
#teY = clf.predict_proba(teX)[:,1]
#save(teY, 'pred_svm05.csv')
