#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import data


def SNHT(x,k):
# Alexandersson, 1986
	return k*np.mean(x[:k])**2 + (len(x)-k)*np.mean(x[k:])**2

	
def maxT(x,k):
# Wang et al., 2007
	N = len(x)
	X1,X2 = np.mean(x[:k]),np.mean(x[k:])
	sk = (np.sum((x[:k]-X1)**2) + np.sum((x[k:]-X2)**2)) / (N-2)
	return ( k*(N-k) / (N * sk) ) **.5 * abs(X1-X2)
	

S = data.benchmark('temp','sur1',0,0)
print sum(np.isnan(S.x))

s = [SNHT(S.x,k) for k in range(1,len(S.x))]
t = [maxT(S.x,k) for k in range(1,len(S.x))]

fig = plt.figure()
ax = plt.subplot()
# plt.plot_date(S.t[1:].astype(date),s,'-')
plt.plot_date(S.t[1:].astype(date),t,'-')
S.plot_vlines(ax,True)
fig.show()

Md = {}
for t,x in zip(S.t.astype(date),S.x):
	try: Md[t.month].append((t.year,x))
	except: Md[t.month] = [(t.year,x)]


fig = plt.figure()
ax = plt.subplot()
for k,v in Md.iteritems():
	y,x = zip(*v)
	d = [date(Y,7,1) for Y in y[:-1]]
	plt.plot_date(d,[maxT(x,j) for j in range(1,len(x))],'-',label=k)
S.plot_vlines(ax,True)
fig.show()