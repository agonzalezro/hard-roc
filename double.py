from re import split
which = 'train.csv'

ip = open(which, 'r')
op = open(which + '.doubled', 'w')

first = True
for line in ip:
  op.write(line.strip() + '\n')
  #print >>op, line

  if first:
    first = False

  else:
    cols = split(r',', line.strip())
    if cols[0] == '0':
      op.write('1,' + ','.join(cols[12:]) + ',' + ','.join(cols[1:12]) + '\n')

    elif cols[0] == '1':
      op.write('0,' + ','.join(cols[12:]) + ',' + ','.join(cols[1:12]) + '\n') 

    else:
      assert False


ip.close()
op.close()
