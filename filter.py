#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


# shift theorem, the following works:
# 	f = np.fft.fft(np.roll(x,m)) 
# is equivalent to 
# 	f = np.fft.fft(x) * np.exp(-2*np.pi*1j*np.fft.fftfreq(len(x)) * m)

# using e.g. A += B with A real and B complex will cast B to real! (for numpy arrays at least)


def up(x,M):
	return np.vstack( (np.atleast_2d(x),np.zeros((M-1,len(x)))) ).T.flatten()[:-M+1]

def ideal(x,N,up=False):
	c = len(x) / (2*N)
	z = len(x) - 2*c
	h = (N if up else 1) * np.hstack((np.ones(c),np.zeros(z),np.ones(c)))
	x = np.real(np.fft.ifft(np.fft.fft(x)*h))
	return x

def _fft(x,a=0,b=0):
	k = np.fft.fftfreq(len(x)).reshape((len(x),1)) + b
	n = np.arange(len(x)).reshape((1,len(x))) + a
	F = np.exp(-2*np.pi*1j * k.dot(n))
	return F.dot(x) * np.exp(-np.pi*1j*k[:,0])
def _ifft(x):
	return _fft(np.roll(x[::-1],1)) / len(x)


def _plot(L):
	fig = plt.figure()
	n = 4
	for i,p in enumerate(L):
		ax = plt.subplot(n*len(L),1,n*i+1)
		ax.plot(p)
		ax.set_xlim((0,len(p)))
		ax = plt.subplot(n*len(L),1,n*i+2)
		ft = _fft(p)
		ax.plot(np.abs(ft))
		ax.plot(np.angle(ft),'-r')
		ax.set_xlim((0,len(p)))
		ax = plt.subplot(n*len(L),1,n*i+3)
		ift = _ifft(ft)
		ax.plot(p-ift)
		ax.set_xlim((0,len(p)))
		ax = plt.subplot(n*len(L),1,n*i+4)
		fift = np.fft.fft(ift)
		ax.plot(np.abs(fift))
		ax.plot(np.angle(fift),'-r')
		ax.set_xlim((0,len(p)))
	fig.show()


def test1():
	L,N = 512,2
	x = 2*np.pi*np.arange(L)/L
	x = np.sin(32*x) + np.sin(200*x)

	y = x[::N]
	# y = ideal(x,N)[::N]
	z = up(y,N)
	z = ideal(up(y,N),N,up=True)
	_plot([x,y,z])
	
def subfft(x,N):
	k = np.fft.fftfreq(len(x)).reshape((len(x),1))
	n = np.arange(len(x)).reshape((1,len(x)))
	F = np.zeros(len(x))
	for m in [0]:#range(N):
		F = F + np.exp(-2*np.pi*1j/N * (k-m).dot(n)).dot(x*(n[0,:]%N==2))
	return F / N

def poly(x,N):
	y = x[::N]
	for n in range(1,N):
		y = np.vstack((y,np.roll(x,-n)[::N]))
	return y

def _pp(x,ax,str='-'):
	plt.plot(x,str)
	ax.set_xlim((0,len(x)))
	ax.axvline(len(x)/2,color='k')

if __name__ == "__main__":
	L = 512
	x = 2*np.pi*np.arange(L)/L
	x = np.sin(50*x) #+ np.sin(200*x)
	
	N = 3
#  	y = np.fft.fft(x[2::N])
	y = subfft(x,N)
# 	f = poly(y,N) * np.array([[1,-1,1]]).T
# 	f = np.sum(f,0)
	f = y
	fig = plt.figure()
	_pp(np.abs(f),plt.subplot(211))
	_pp(np.angle(f),plt.subplot(212),'-r')
# 	_pp(np.imag(f),plt.subplot(313),'-r')
	fig.show()
