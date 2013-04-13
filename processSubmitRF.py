import os
import sys
FH=open('/Users/adityajitta/Desktop/subRFCombine3.csv')# output file from rfBigData.r
FH1=open('/Users/adityajitta/Desktop/subRFCombineR3.csv','w+')# Submission file name
count=0
for line in FH.readlines():
	line=line.strip()
	tk=line.split(',')
	first=tk[-1]
	val=[tk[temp] for temp in range(len(tk)-1)]
	sval=','.join(val)
	tdata=first+','+sval+'\n'
	FH1.write(tdata)
FH.close()
FH1.close()