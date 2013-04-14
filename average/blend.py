import glob
from numpy import loadtxt, savetxt 

def wcl(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass

    return i + 1

blended = None
files = glob.glob('0*.csv')
num_files = len(files)

for file in files:
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

  if blended == None:
    blended = this[:,0] / num_files

  else:
    blended = blended + this[:, 0] / num_files

print num_files
savetxt('blended.csv', blended, fmt = '%.5f')
