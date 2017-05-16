#!/usr/bin/env python
import sys
sys.path.append('../DMI/db/')
import matrices as mat
import numpy as np
from datetime import date
from schema import Station, Session



def station_breaks(id):
	breaks = []
	for s in Session.query(Station).filter(Station.station_id==id).order_by(Station.startdate):
		try: breaks.append((s.startdate,s.enddate))
		except: breaks.append((s.startdate,))
	return breaks

def break_plot(id,ax):
	for b in station_breaks(id):
		ax.axvline(b[0],color='g')
		try: ax.axvline(b[1],color='r')
		except: pass

class Breaks(object):
	def __init__(self,query,r,t):
		self.parts = []
		self.means = []
		self.hist = []
		a = 0
		for s in query:
			try: i = [j for j,d in enumerate(t) if d>=s.startdate and d<=s.enddate]
			except TypeError: i = np.where(t>=s.startdate)[0]
			if i[0]>a: 
				self._append(r[a:i[0]])
			self._append(r[i])
			a = i[-1] + 1
		if i[-1]<len(r)-1:
			self._append(r[i[-1]+1:])
	
	def _append(self,x):
		self.parts.append(x)
		self.means.extend(np.nanmean(x)*np.ones(x.shape))
		self.hist.append(np.histogram(x))
		
def plot_hist(B):
	p = 1
	fig = plt.figure()
	for j in B:
		for i,b in enumerate(j.hist):
			ax = plt.subplot(len(B),len(j.hist),p)
			ax.bar(b[1][:-1],b[0],np.diff(b[1]))
			ax.axvline(np.nanmean(j.parts[i]),color='r')
			p += 1
	fig.show()


def Gauss(sigma):
	w = sigma*3
	x = np.arange(-w,w+1,dtype='float')
	f = np.exp(-x**2/(2*sigma**2))
	return f/np.sum(f)

def dGauss(sigma):
	w = sigma*3
	x = np.arange(-w,w+1,dtype='float')
	f = np.exp(-x**2/(2*sigma**2)) * x/sigma**2
	return f/np.sum(f**2)**.5

class Filter(object):
	def __init__(self,x,omega=10,iter=1):
		i = np.isnan(x)
		x[i] = 0
		filt = np.ones(x.shape)
		filt[omega:-omega+1] = 0.
		self.f = np.fft.fft(x) * filt
		self.out = np.real(np.fft.ifft(self.f)).reshape((1,x.shape[0]))
		for j in range(iter-1):
			z = x - self.out[-1,:]
			z[i] = 0
			self.f += np.fft.fft(z) * filt
			self.out = np.vstack((self.out, np.real(np.fft.ifft(self.f))))
			
	def apply(self,filt):
		n = self.f.shape[0] + filt.shape[0] - 1
		l = filt.shape[0]/2
		f = np.hstack((self.f,np.zeros(filt.shape[0]-1)))
		return np.real(np.fft.ifft((f*np.fft.fft(filt,n)))[l:-l])


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	st = Session.query(Station).filter(Station.station_id==4360).order_by(Station.startdate)
	R = mat.Session.query(mat.Matrix).get(4).dbmatrix
	t = R.cols.astype(date)
	r = R[65,:]
	
	f = Filter(r,omega=30,iter=1)
	
	fig = plt.figure()
	ax = plt.subplot()
	break_plot(4360,ax)
	B = [Breaks(st,r,t)]
# 	plt.plot(t,r)
	plt.plot(t,B[0].means)

	for w in [182,365,730]:
		try: g = np.vstack((g,f.apply(dGauss(w))))
		except NameError: g = f.apply(dGauss(w))
	for i in range(g.shape[0]):
		plt.plot(t,g[i,:])
		B.append(Breaks(st,g[i,:],t))
		plt.plot(t,B[-1].means)
	fig.show()