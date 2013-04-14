import glob
from numpy import loadtxt, savetxt, array

def wcl(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass

    return i + 1

blended = None
files = glob.glob('*.csv')
num_files = len(files)

aucs = []
auc_files = glob.glob('*.auc')
for file in auc_files:
  scalar = loadtxt(file)
  aucs.append(scalar)

from numpy import min
aucs = array(aucs)
aucs = (array(aucs)-min(aucs)+0.001)
Z = sum(aucs)
aucs = aucs/Z
print aucs

for (auc, file) in zip(aucs, files):
  num_lines = wcl(file)
  if num_lines ==  5953:
    this = loadtxt(file, delimiter=',', skiprows=1)

  elif num_lines == 5952:
    this = loadtxt(file, delimiter=',')

  else:
    assert False

  if this.ndim == 1:
    this = this.reshape(5952, 1)

  else:
    this = this[:,0]
    this = this.reshape(5952, 1)

  print auc
  if blended == None:
    blended = this[:,0] * auc

  else:
    blended = blended + this[:, 0] * auc

print num_files
savetxt('blended.csv', blended, fmt = '%.5f')
