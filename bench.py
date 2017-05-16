#!/usr/bin/env python
import csv, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date


class series(object):
	def __init__(self,name,var,type,n):
		self.breaks = []
		self.outliers = []
		t = []
		for h,s in (('inho','ratnm'),('orig','hotnm')):
			d = []
			fname = '{}{}.txt'.format(s,name)
			path = os.path.join('benchmark',h,var,type,n)
			with open(os.path.join(path,fname)) as file:
				for r in csv.reader(file,delimiter='\t'):
					d.extend([float(x) for x in r[1:13]])
					if h=='orig':
						t.extend(['{}-{:02d}'.format(r[0],m) for m in range(1,13)])
			setattr(self,h,np.array(d))
			self.t = np.array(t,dtype='datetime64[M]')
		with open(os.path.join(path,'{}detected.txt'.format(n))) as file:
			for r in csv.reader(file,delimiter='\t'):
				if r[0]==fname:
					if r[1]=='OUTLIE':
						self.outliers.append(date(int(r[2]),int(r[3]),1))
					elif r[1]=='BREAK':
						self.breaks.append(date(int(r[2]),int(r[3]),15))
	
	def plot_vlines(self,ax):
		for l in self.breaks:
			ax.axvline(l,color='g')
		for l in self.outliers:
			ax.axvline(l,color='r')


def fftconv(x,f):
	l = x.shape[0]+f.shape[0]-1
	return np.real(np.fft.ifft(np.fft.fft(x,l)*np.fft.fft(f,l)))


def Gauss(sigma):
	w = sigma*3
	x = np.arange(-w,w+1,dtype='float')
	f = np.exp(-x**2/(2*sigma**2))
	return f/np.sum(f)

def dGauss(sigma):
	w = sigma*3
	x = np.arange(-w,w+1,dtype='float')
	f = np.exp(-x**2/(2*sigma**2)) * x/sigma**2
	return f

def _plot(s,F):
	fig = plt.figure()
	ax = plt.subplot()
# 	plt.plot(s.inho-s.orig,'-')
	for f in F:
		plt.plot(f,'-')
	for l in s.breaks:
		ax.axvline(l+.5,color='k')
	for l in s.outliers:
		ax.axvline(l,color='r')
	ax.axhline(0)
	fig.show()


if __name__ == "__main__":
	var = 'temp'
	type = 'sur1'
	s = series('41097001d',var,type,'000002')
	
	G = Gauss(24)
	l = G.shape[0]/2
	
	f = np.convolve(s.inho,G,'same')
	g = fftconv(s.inho,G)[l:-l]
	_plot(s,[f,g])
	
	
# 	g = [np.convolve(s.inho,dGauss(w),'same') for w in [12,24,48]]
# 	g = np.vstack((np.array(g),g[0]))
# 	for i in range(1,3):
# 		g[3,:] *= g[i,:]
# 	_plot(s,g)
	
	