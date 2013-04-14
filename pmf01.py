from numpy import array, float64, ones, zeros, dot
from numpy.random import rand
import beta_ntf
from counts import adj
from data import draft, eval


trX, trY, teX = draft()
w = [8]
identities, counts, total = adj(trX, trY, teX, w)

N = len(counts.keys())

W = zeros((N,N))
X = rand(N,N)

for i in range(trY.shape[0]):
  l = identities[i][0]-1
  r = identities[i][1]-1
  X[l, r] = trY[i]
  W[l, r] = 1

eps = 0.01
X = X + rand(N,N) * eps
W = W + rand(N,N) * eps
ntf = beta_ntf.BetaNTF(X.shape, n_components=2, beta=10.0, n_iter=1000,verbose=True)

ntf.fit(X, W)
U = ntf.factors_[0]
V = ntf.factors_[1]

teY = []
for i in range(trY.shape[0], trY.shape[0]+teX.shape[0]):
  l = identities[i][0]-1
  r = identities[i][1]-1
  teY.append(dot(U[l,:], V[r,:].T))

teY = array(teY)

print eval(teY)
