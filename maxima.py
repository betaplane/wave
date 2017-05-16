#!/usr/bin/env python
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from schema import *
import matplotlib.pyplot as plt
from threading import Thread
from Queue import Queue
import pandas as pd

	
def series(field):
	x,ym = zip(*[(float(r.x)*float(field.mult),(r.t.year,r.t.month)) for r in field.records])
	return pd.Series(x,index=pd.MultiIndex.from_tuples(ym, names=['Year','Month']),name=field.station_id)


def mtrace(W,P,N):
	x = np.r_[P[0],N[0]]
	y = np.argmax(np.abs(W[x]))
	M = N if y>=len(P) else P
	m = x[y]
	for l in M[1:]:
		m = l[np.argmin([np.abs(m-j) for j in l])]
	return m

def CWT(x,J):
	S = np.fft.fft(np.r_[x,x[::-1]])
	f = 2 * np.pi * np.fft.fftfreq(2*len(x))
	P,N = DWT(x,2)
	for j in range(2,J):
		w = (j+1) * f
		W = np.real(np.fft.ifft(S * 1j*w * np.sinc(w/(4*np.pi))**4 ))[:len(x)]
		P.insert(0,sig.argrelmax(W)[0])
		N.insert(0,sig.argrelmin(W)[0])
	return mtrace(W,P,N)

def SWT(x,j):
	S = np.fft.fft(np.hstack((x,x[::-1])))
	f = 2 * np.pi * np.fft.fftfreq(2*len(x))	
	w = (j+1) * f
	W = np.real(np.fft.ifft(S * 1j*w * np.sinc(w/(4*np.pi))**4 ))[:len(x)]
	return sig.argrelmax(W)[0],sig.argrelmin(W)[0]

def DWT(x,J):
	g = (-2.,2.)
	h = (.125,.375,.375,.125)
	
	def conv(x,h):
		return sig.convolve(x,h[::-1],'same')
	
	def up(x):
		return np.r_['0,2',x,np.zeros(len(x))].T.flatten()[:-1]

	P,N = [],[]
	S = np.roll(np.r_[x,x[::-1]],len(x)/2)
	for j in range(J):
		W = np.roll(conv(S,g),-len(x)/2)[:len(x)]
		S = conv(S,h)
		h,g = [up(f) for f in [h,g]]
		P.insert(0,sig.argrelmax(W)[0])
		N.insert(0,sig.argrelmin(W)[0])
	return P,N
# 	return mtrace(W,P,N)


class _thread(Thread):
	def __init__(self,queue):
		Thread.__init__(self)
		self.queue = queue
		self.I = []
	def run(self):
		while True:
			bp,x = self.queue.get()
# 			i = DWT(x,7)
			i = CWT(x,2**9)
			self.I += [(i-bp)]
			print self.queue.qsize()
			self.queue.task_done()

def simulate(x,nthreads=20):
	N = x.shape[1]
	w = np.random.normal(0,np.std(x),x.shape)
	threads = []
	q = Queue()
	for t in range(nthreads):
		th = _thread(q)
		th.setDaemon(True)
		th.start()
		threads += [th]
	for j in range(x.shape[0]):
		bp = np.random.randint(.2*N,.8*N)
		y = np.hstack((x[j,:bp],x[j,bp:]+2.))
		q.put((bp,y))
	q.join()
	I = []
	for th in threads[:]:
		I += th.I
	return I

def _hist(x,n=2000):
	def _text(s,ax):
		bb = ax.get_position()
		x = bb.x0+0.01
		fig.text(x,bb.y1-0.01,s,size=16,weight='bold',va='top',ha='left')
	
	fig = plt.figure()
	bins = np.r_[-n,np.arange(-12.5,13),n]
	ax = plt.subplot()
	h,b = np.histogram(x,bins)
	plt.bar(b[1:-2],h[1:-1],width=1)
	ax.axvline(0,color='r',lw=2)
	_text('{}'.format(h[0]+h[-1]),ax)
	ax.set_xlim((-12.5,12.5))	
	ax.set_title('SIAAFT$^{**}$ surrogate')
# 	fig.tight_layout()
	fig.show()
	return fig
	
if __name__ == "__main__":
# 	f = Session.query(Field).get(2418)
	
# 	for f in q[:1]:
# 	x = np.array([float(r.x)*float(f.mult) for r in f.records])
# 	x = sio.loadmat('../AGU2015/surrogate.mat')['x']
# 	I = simulate(x)
# 	sio.savemat('traces_512_CWT+DWT.mat',{'I':I})
# 	I = sio.loadmat('traces.mat')['I']
# 	_hist(I)
	
# 	x = np.r_[np.zeros(500),np.ones(500)]
# 	w = SWT(x,2**9)
	
# 	D = pd.DataFrame()
# 	q = Session.query(Field).filter(Field.source=='DMI_subd',Field.name=='t month')
# 	for f in q:
# 		D = pd.concat((D,series(f)),axis=1)
	

	for d in D:
		i = np.isnan(D[d]).astype(int)
		j = np.r_[0,np.where(np.diff(i))[0]+1,len(i)]
		for k in np.where(np.diff(j)>=150)[0]:
			D[d]
			