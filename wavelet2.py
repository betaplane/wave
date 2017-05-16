#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import date
from scipy import linalg




def _plot(T,L,S):
	J = len(L[0].a)
	A = lambda x:x#np.abs(x)
	t1 = np.arange(len(S.x))
	# half-integer sampling shift
	t2 = t1-.5
# 	t2 = (T - np.diff(np.hstack((T[0]-np.timedelta64(1,'M'),np.array(T,dtype='datetime64[D]'))))/2).astype(date)
	t3 = t1
	
	fig = plt.figure()
	# diff original - reconstruction
	try:
		ax = plt.subplot(J+1,2,1)
		plt.plot(S.x-L[0].r[0],'-g')
		ax.set_xticklabels([])
		ax = plt.subplot(J+1,2,2)
		# reconstruction is shifted by full-integer step
		plt.plot(S.x[:-1]-L[1].r[0][1:],'-m')
		ax.set_xticklabels([])
	except: pass
	
	
	for j in range(J):
		ax1 = plt.subplot(J+1,2,2*j+3)
		ax2 = plt.subplot(J+1,2,2*j+4)
	# analysis
		# "approximation"
		ax1.plot(t1,L[0].a[j],'-')
		ax1.plot(t2,L[1].a[j],'-r')
		t4 = t3[::2]-2**(j-1)
		ax1.plot(t4,L[3].a[j],'-c')
		
		# "detail" (wavelet)
		ax2.plot(t1,A(L[0].w[j]),'-')
		ax2.plot(t2,A(L[1].w[j]),'-r')
		ax2.plot(t1,A(L[2][j,:]),'-g')
		ax2.plot(t3-2**(j-1),A(L[3].w[j]),'-c')
		t3 = t4
		
		ax1.set_xlim((0,1199))
		ax2.set_xlim((0,1199))
		if j<J-1:
			ax1.set_xticklabels([])
			ax2.set_xticklabels([])
	# synthesis
# 			ax1.plot(t1,L[0].r[j],'-g')
# 			ax1.plot(t2,L[1].r[j],'-m')
			
		S.plot_vlines(ax1,date=False)
		S.plot_vlines(ax2,date=False)
		
	fig.show()

def scales_plot(S,F,P):
	J = len(F.a)
	A = lambda x:np.abs(x)
	t1 = np.arange(len(S.x))
	t2 = t1
	
	lc = 'grey'
	mc = 'cyan'
	
	fig = plt.figure(figsize=(12,9))

	for j in range(J):
		ax1 = plt.subplot(J,2,2*j+1)
		ax2 = plt.subplot(J,2,2*j+2)
	# analysis
		# "detail" (wavelet)
		ax2.plot(t1,A(F.w[j]),ls='-',color=lc)
		t2 = t2-2**(j-1)
		if j>3:
			ax2.plot(t2,A(P.w[j]),ls='',marker='o',color=mc)
		
		# "approximation"
		ax1.plot(t1,F.a[j],ls='-',color=lc)
		t2 = t2[::2]
		if j>3:
			ax1.plot(t2,P.a[j],ls='',marker='o',color=mc)
		
		ax1.set_xlim((0,len(S.x)))
		ax2.set_xlim((0,len(S.x)))
		if j<J-1:
			ax1.set_xticklabels([])
			ax2.set_xticklabels([])
			
		plot_vlines(ax1,S,True)
		plot_vlines(ax2,S,True)
		
		ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
		ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
		for i in (ax1.get_yticklabels() + ax2.get_yticklabels()):
			i.set_fontsize(10)
		
	fig.show()
	return fig


class WT(object):
	def __init__(self,S,J):
		self.w,self.a,self.r,self.J,self.t = [],[],[],J,[S.t]
		self.analysis(S.x)
		self.synthesis()
	@classmethod
	def up(cls,x,n=2,N=0):
		v = np.vstack( (np.atleast_2d(x),np.zeros((n-1,len(x)))) ).T.flatten()
		return v[:N] if N else v[:-n+1]
	@classmethod
	# correction for numerical error at small scales
	def _la(cls,j):
		return (1.5,1.12,1.03,1.01)[j] if j<4 else 1.


class DWT_fft(WT):
	def analysis(self,x):
		N = len(x)
		self.S = np.fft.fft(np.hstack((x,x[::-1])))
		f = np.pi * np.fft.fftfreq(2*N)
		for j in range(self.J):
			self.w.append( np.fft.ifft(self.S * 4*1j*np.sin(f))[:N] / self._la(j) )
			self.S = self.S * np.cos(f)**3
			self.a.append( np.fft.ifft(self.S)[:N] )
			f *= 2.
	
	def synthesis(self):
		f = 2**(self.J-1) * np.pi * np.fft.fftfreq(len(self.S))
		for j in range(self.J)[::-1]:
			m = self.S[0]
			H = np.cos(f)**3
			self.S = self.S * np.conj(H)
			K = (1 - H**2) / (4*1j * np.sin(f))
			self.S += self._la(j) * np.fft.fft(np.hstack((self.w[j],self.w[j][::-1]))) * K
			self.S[0] = m
			self.r.insert(0,np.fft.ifft(self.S)[:len(self.S)/2])
			f /= 2.

class DWTC(WT):
	g = (-2.,2.)
	h = (.125,.375,.375,.125)
	k = (.0078125,.054685,.171875,-.171875,-.054685,-.0078125)
	
	def conv(self,x,h,circ=False):
		l = (len(h))/2
		y = np.hstack((x,x[::-1]))
		return np.roll(np.convolve(y,h[::-1],'valid'),l)[:len(x)]


class DWT_conv(DWTC):
	def analysis(self,x):
		S = x
		for j in range(self.J):
			h,g = self.filt([self.h,self.g],2**j)
			self.w.append( self.conv(S,g) / self._la(j) )
			S = self.conv(S,h)
			self.a.append(S)
		
	def filt(self,F,s):
		return [self.up(f,s) for f in F] if s>1 else F
				
	def synthesis(self):
		S = self.a[-1]
		for j in range(self.J)[::-1]:
			h,k = self.filt([self.h[::-1],self.k],2**j)
			S = self.conv(S,h)
			S += self.conv(self.w[j],k) * self._la(j)
			self.r.insert(0,S)

class DWT_pars(DWTC):
	def analysis(self,x):
		S = x
		for j in range(self.J):
			w = self.conv(S,self.g) / self._la(j)
			self.w.append( w )
			S = self.conv(S,self.h)[::2]
			self.a.append(S)
			self.t.append(self.t[-1][::2])
	
	def synthesis(self):
		S = self.a[-1]
		h = self.h[::-1]
		for j in range(self.J)[::-1]:
			S = 2 * self.conv(self.up(S,N=len(self.w[j])),h)
			S += self.conv(self.w[j],self.k) * self._la(j)
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
	

def interp(x,t):
	N = len(x)
	i = np.flatnonzero(~np.isnan(x))
	x[np.isnan(x)] = 0
	S = np.fft.fft(np.hstack((x,x[::-1])))
	S[N/t:-N/t] = 0
	s = np.real(np.fft.ifft(S))[:N]
	for j in range(4):
		z = s
		S[i] = x[i]-s[i]
		S[N:] = S[:N][::-1]
		S = np.fft.fft(S)
		S[N/t:-N/t] = 0
		s += np.real(np.fft.ifft(S))[:N]
		print np.sum((s-z)**2)**.5
	return s
		

def plot_vlines(ax,S,diff=False):
	for b in S.breaks:
		if b.type=='break':
			if diff:
				c = 'r' if b.jump>0 else 'b'
				ax.axvline(b.index+.5,color=c,lw=2)
			else: ax.axvline(b.index+.5,color='lime',lw=2)
	
	
if __name__ == "__main__":
	import data
	from wavelet import maxmod
	S = data.benchmark('temp','sur1',0,0)
# 	S = data.DMI(4360)
	
	J = 8
	F = DWT_fft(S,J)
# 	C = DWT_conv(S,J)
	W = CWT(S.x,2**J)
	P = DWT_pars(S,J)
# 	_plot(S.t,[F,C,W[2**np.arange(1,J+1)-1,:],P],S)
	
# 	fig=plt.figure()
# 	ax = plt.subplot()
# 	plt.contour(W,[0])
# 	S.plot_vlines(ax,False)
# 	fig.show()
	
	
	fig1 = scales_plot(S,F,P)
	
# 	mm = maxmod(W)
# 
# 	fig=plt.figure(figsize=(12,6))
# 	ax = plt.subplot()
# 	plt.pcolor(W,cmap=plt.get_cmap('coolwarm'),vmin=-1.5,vmax=1.5)
# 	for a in mm.curves[:10]: 
# 		plt.plot(a,range(len(a)),'w',lw=2)
# 	ax.set_ylim((0,2**J))
# 	ax.invert_yaxis()
# 	ax.xaxis.tick_top()
# 	ax.set_ylabel('scale')
# 	ax.set_xlim((0,len(S.x)))
# 	plot_vlines(ax,S)
# 	fig.show()


	
	