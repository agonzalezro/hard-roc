from numpy import array, float64, ones, zeros, dot, hstack
from numpy.random import rand
import beta_ntf
from counts import adj
from data import draft, eval, proof


def new_features(trX, trY, teX):
  w = [0,1]
  identities, counts, total = adj(trX, trY, teX, w)

  N = len(counts.keys())

  eps = 0.0001
  W = rand(N,N)*eps
  X = rand(N,N)

  for i in range(trY.shape[0]):
    l = identities[i][0]-1
    r = identities[i][1]-1
    X[l, r] = trY[i] + rand() * eps
    W[l, r] = 1.0

  ntf = beta_ntf.BetaNTF(X.shape, n_components=10, beta=2, n_iter=10,verbose=True)

  #ntf.fit(X)
  ntf.fit(X, W)
  U = ntf.factors_[0]
  V = ntf.factors_[1]

  return hstack((U, V))

