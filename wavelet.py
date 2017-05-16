#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def up(x,n=2,N=0):
	v = np.vstack( (np.atleast_2d(x),np.zeros((n-1,len(x)))) ).T.flatten()
	return v[:N] if N else v[:-n+1]

def _plot(L,S,scales):
	fig = plt.figure()
	J = len(scales)
	for i,j in enumerate(scales):
		ax = plt.subplot(J+1,2,2*i+1)
		try: plt.plot_date(L[0].t(j+1),L[0].a[j],'-')
		except: pass
		try: plt.plot_date(L[1].t(j+1),L[1].a[j],'-r')
		except: pass
		if scales[-1]-j>1: 
			try: plt.plot_date(L[0].t(j+1),L[0].r[j+1],'-g')
			except: pass
			try: plt.plot_date(L[1].t(j+1),L[1].r[j+1],'-m')
			except: pass
		S.plot_vlines(ax)
		
		A = lambda x:np.abs(x)
		ax = plt.subplot(J+1,2,2*i+2)
		plt.plot_date(L[0].t(j),A(L[0].w[j]),'-')
		plt.plot_date(L[1].t(j),A(L[1].w[j]),'-r')
		S.plot_vlines(ax)

	ax = plt.subplot(J+1,2,2*J+1)
	try: plt.plot_date(L[0]._t,S.x-L[0].r[0],'-g')
	except: pass

	ax = plt.subplot(J+1,2,2*J+2)
	try: plt.plot_date(L[1]._t,S.x-L[1].r[0],'-m')
	except: pass
	fig.show()



class WT(object):
	def __init__(self,S,J):
		from datetime import date
		self._t,self.J = S.t.astype(date),J
		self.w,self.a,self.r,self.j = [],[],[],[]
		self.analysis(S.x)
		try: self.synthesis()
		except: pass
		
	def t(self,j):
		return self._t[::self.j[j]]
		
	@classmethod
	def _la(cls,j):
		return (1.5,1.12,1.03,1.01)[j] if j<4 else 1.


class DWT_fft(WT):
	def _fft(self,x,a=0,b=0):
		return np.fft.fft(x)
		k = np.fft.fftfreq(len(x)).reshape((len(x),1)) + b
		n = np.arange(len(x)).reshape((1,len(x))) + a
		F = np.exp(-2*np.pi*1j * k.dot(n))
		return F.dot(x)
	def _ifft(self,x,a=0,b=0):
		return np.fft.ifft(x)
		return self._fft(np.roll(x[::-1],1),a,b) / len(x)
	
	def analysis(self,x):
		self.j = np.ones(self.J+1)
		f = np.pi * np.fft.fftfreq(len(x))
		S = self._fft(x)
		for j in range(self.J):
			e = 1.#np.exp(1j*f)
			self.w.append( self._ifft(S * 4*1j*e*np.sin(f)) / self._la(j) )
			S = S * e*np.cos(f)**3
			self.a.append( self._ifft(S) )
			f *= 2.
	
	def synthesis(self):
		N = len(self.w[0])
		f = 2**(self.J-1) * np.pi * np.fft.fftfreq(N)
		S = self._fft(self.a[-1])
		for j in range(self.J)[::-1]:
			m = S[0]
			e = 1.#np.exp(1j*f)
			H = e * np.cos(f)**3
			S = S * np.conj(H)
			K = (1 - np.abs(H)**2) / (4*1j*e * np.sin(f))
			S += self._la(j) * self._fft(self.w[j]) * K
			S[0] = m
			self.r.insert(0,self._ifft(S))
			f /= 2.

class DWTC(WT):
	g = (-2.,2.)
	h = (.125,.375,.375,.125)
	k = (.0078125,.054685,.171875,-.171875,-.054685,-.0078125)
	
	def mult():
		pass
	
	def conv(self,x,h,circ=False):
		l = (len(h))/2
		y = np.hstack((x,x[::-1]))
		if not circ:
			return np.roll(np.convolve(y,h[::-1],'valid'),l)[:len(x)]
		else:		
			z = np.zeros(len(y))
			z[:l+1],z[-l:] = h[l:],h[:-l-1]
			return linalg.circulant(y).dot(z)[:len(x)]

class DWT_conv(DWTC):
	def analysis(self,x):
		self.j = 2**np.arange(self.J+1)
		S = x
		for j in range(self.J):
			w = self.conv(S,self.g) / self._la(j)
			self.w.append( w )
			S = self.conv(S,self.h)[::2]
			self.a.append(S)
	
	def synthesis(self):
		S = self.a[-1]
		h = self.h[::-1]
		for j in range(self.J)[::-1]:
			S = 2 * self.conv(up(S,N=len(self.w[j])),h)
			S += self.conv(self.w[j],self.k) * self._la(j)
			self.r.insert(0,S)

class DWT_conv_ns(DWTC):
	def analysis(self,x):
		self.j = np.ones(self.J+1)
		S = x
		for j in range(self.J):
			h,g = self.filt([self.h,self.g],2**j)
			self.w.append( self.conv(S,g) / self._la(j) )
			S = self.conv(S,h)
			self.a.append(S)
	
	def filt(self,F,s):
		return [up(f,s) for f in F] if s>1 else F
				
	def synthesis(self):
		S = self.a[-1]
		for j in range(self.J)[::-1]:
			h,k = self.filt([self.h[::-1],self.k],2**j)
			S = self.conv(S,h)
			S += self.conv(self.w[j],k) * self._la(j)
			self.r.insert(0,S)


def CWT(x,J):
	N = len(x)
	S = np.fft.fft(np.hstack((x,x[::-1])))
	f = 2 * np.pi * np.fft.fftfreq(2*N)
	M = np.zeros((J,N))
	for j in range(J):
		w = (j+1) * f
		W = 1j*w * np.sinc(w/(4*np.pi))**4
		M[j,:] = np.real(np.fft.ifft(S * W)[:N])
	return M


class maxmod(object):
	def __init__(self,W):
		self.wc = W.copy()
		try: W = [W[i,:] for i in range(W.shape[0])]
		except: pass
		self.mean,self.curves,self.sign = [],[],[]
		for w in W:
			u = np.abs(w)
			a,b = [set((1+np.where(u[1:-1]-v>=0)[0]).tolist()) for v in (u[:-2],u[2:])]
			c,d = [set((1+np.where(u[1:-1]-v>0)[0]).tolist()) for v in (u[:-2],u[2:])]
			i = list((a&b) & (c|d))
			self.mean.append(np.mean(u[i]))
			s = lambda x,i:2*x[i]-x[i-1]-x[i+1]
			p,n = [j for j in i if s(w,j)>0],[j for j in i if s(w,j)<0]
			try:
				MP,P = self._trace(MP,P,p)
				MN,N = self._trace(MN,N,n)
			except UnboundLocalError:
				MP = np.atleast_2d(sorted(p))
				MN = np.atleast_2d(sorted(n))
				P,N = [],[]
		self._extend([MP[:,k] for k in range(MP.shape[1])],1)
		self._extend([MN[:,k] for k in range(MN.shape[1])],-1)
		self._extend(P,1)
		self._extend(N,-1)
		j = np.argsort([len(c) for c in self.curves])[::-1]
		self.curves = [self.curves[i] for i in j]
		self.sign = [self.sign[i] for i in j]
	
	def _extend(self,X,sign):
		self.curves.extend(X)
		self.sign = np.hstack( (self.sign, sign*np.ones(len(X))) )
	
	def _trace(self,Act,Pass,new):
		k = [np.argmin([abs(n-j) for j in Act[-1,:]]) for n in new]
		l = set(range(Act.shape[1])) - set(k)
		Pass.extend([Act[:,j] for j in l])
		Act = np.vstack((Act[:,k],new))
		return Act,Pass
	



class mplot(object):
	def __init__(self,M,S):	
		from matplotlib.widgets import Button
		self.fig = plt.figure()
		self.artists = []
		self.ax1,self.ax2 = plt.subplot(121), plt.subplot(122)
		self.p1,self.p2,self.C = [],[],[]
		for a in M.curves[:20]:
			self.p1.append( self.ax1.plot(a,range(len(a)),'k',picker=10)[0] )
			self.C.append( [M.wc[i,a[i]] for i in range(len(a))] )
			self.p2.append( self.ax2.plot(np.abs(self.C[-1]),'k',picker=10)[0] )
		S.plot_vlines(self.ax1,False)
		p = self.ax2.get_position()
		self.B = Button(plt.axes([p.x1-.1,p.y1-.1,.08,.05]),'log')
		self.B.on_clicked(self.logscale)
		self.fig.canvas.mpl_connect('pick_event', self.pick)
		self.fig.show()
	
	def logscale(self,e):
		if self.ax2.get_xscale()=='linear':
			y = self.ax2.get_ylim()[1]
			self.ax2.set_xscale('log')
			self.ax2.set_yscale('log')
			self.alpha1 = self.ax2.plot((1,y),(1,y),'-c')[0]
		else:
			self.ax2.set_xscale('linear')
			self.ax2.set_yscale('linear')
			self.alpha1.remove()
		self.fig.canvas.draw()
	
	def pick(self,e):
		try: 
			i = self.p1.index(e.artist)
			A = (e.artist,self.p2[i])
		except: 
			i = self.p2.index(e.artist)
			A = (e.artist,self.p1[i])
		print i
		if e.artist in self.artists:
			for a in A:
				plt.setp(a,color='k',linewidth=1)
				self.artists.remove(a)
		else:
			for a in A:
				plt.setp(a,color='r',linewidth=2)
				self.artists.append(a)
		self.fig.canvas.draw()
	

def spect(S):
	fig=plt.figure()
	ax = plt.subplot()
	plt.plot(np.abs(np.fft.fft(S.x)))
	ax.set_xscale('log')
	ax.set_yscale('log')
	fig.show()

def hist(x):
	hist, bins = np.histogram(x, bins=np.arange(min(x),max(x)+2)-.5)
	f = plt.figure()
	plt.bar(bins[:-1], hist, width=1)
	f.show()

	
	
if __name__ == "__main__":
# 	x = np.zeros(1200)
# 	x[600:] = 1.
# 	t = np.arange('1990-01-01','2000-01-01',dtype='datetime64')[:1200]
# 	T = (t[600]- np.timedelta64(hours=12)).astype(date)
# 	t = t.astype(date)
# 	lines = lambda ax:ax.axvline(T)

	import data
	S = data.benchmark('temp','sur1',0,0)
# 	S = data.DMI(4360)
	
	J = 8
	F = DWT_fft(S,J)
	C = DWT_conv(S,J)
	_plot([F,C],S,range(J))
	
# 	s = 300
# 	W = CWT(S.x,s)
# 	M = maxmod(W)
# 	W[:15,:] = 0
# 
# 	fig=plt.figure()
# 	ax = plt.subplot()
# 	plt.pcolor(W)
# 	for a in M.curves[:20]:
# 		plt.plot(a,range(len(a)),'w')
# 	S.plot_vlines(ax,False)
# 	fig.show()
# 
# 	p = mplot(M,S)

# 	Cm = np.array([C[i,:]*C[i+1] for i in range(C.shape[0]-1)])
# 	fig=plt.figure()
# 	ax = plt.subplot()
# 	plt.pcolor(Cm[15:,:])
# 	S.plot_vlines(ax,False)
# 	fig.show()