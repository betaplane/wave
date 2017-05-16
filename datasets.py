#!/usr/bin/python
import os, csv, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpld
from datetime import date
from matplotlib.ticker import MaxNLocator
import scipy.io as sio
import scipy.signal as sig


def data(n=1):
	path = 'benchmark/orig/temp/sur1'
	L = []
	for dir in [d for d in os.listdir(path) if d[0]!='.']:
		p = os.path.join(path,dir)
		for file in [f for f in os.listdir(p) if f[0]!='0' and f[0]!='.']:
			t,x = [],[]
			fp = os.path.join(p,file)
			with open(fp) as f:
				for r in csv.reader(f,delimiter='\t'):
					t.extend(['{}-{:02d}'.format(r[0],m) for m in range(1,13)])
					x.extend([float(s) for s in r[1:13]])
			x = np.array(x)
			if not np.where(np.diff(np.where(x==-999.9)[0])>1)[0].shape[0]:
				x = x[np.where(x!=-999.9)[0]]
				N = x.shape[0]
				bp = np.random.randint(.1*N,.9*N)
# 				bs = np.random.normal(0,.8,1)[0]
				for i in range(n):
					y = np.hstack((x[:bp],x[bp:]+1.))
					L.append([x,y,bp])
	return L
		

def CWT(x,J):
	N = len(x)
	S = np.fft.fft(np.hstack((x,x[::-1])))
	f = 2 * np.pi * np.fft.fftfreq(2*N)
	M = np.zeros((len(J),N))
	for i,j in enumerate(J):
		w = (j+1) * f
# 		W = np.exp(1j*w/2.) * 1j*w * np.sinc(w/(4*np.pi))**4
		W = 1j*w * np.sinc(w/(4*np.pi))**4
		M[i,:] = np.real(np.fft.ifft(S * W)[:N])
	return M

	
def _plot(l):
	s = range(257)
	c = CWT(l[1],s)
	I = trace(c)
	T,mT = maxT(l[1])
	W,mW = Wilcoxon(l[1])
	m,v = slope(c[80:200,])
	fig = plt.figure()
	ax = plt.subplot2grid((9,1),(0,0))
	plt.plot(T)
	ax.axvline(l[2]-.5,color='g')
	ax.axvline(mT+.5,color='m')
	ax = plt.subplot2grid((9,1),(1,0))
	plt.plot(W)
	ax.axvline(l[2]-.5,color='g')
	ax.axvline(mW+.5,color='m')
	ax = plt.subplot2grid((9,1),(2,0))
	plt.plot(np.abs(c[256,:]))
	ax.axvline(l[2]-.5,color='g')
	ax.axvline(m,color='m')
	ax.axvline(v,color='r')
	ax = plt.subplot2grid((9,1),(3,0))
	plt.plot(np.abs(c[128,:]))
	ax.axvline(l[2]-.5,color='g')
	ax.axvline(m,color='m')
	ax.axvline(v,color='r')
	ax = plt.subplot2grid((9,1),(4,0))
	plt.plot(np.abs(c[64,:]))
	ax.axvline(l[2]-.5,color='g')
	ax.axvline(m,color='m')
	ax.axvline(v,color='r')
	ax = plt.subplot2grid((9,1),(5,0),rowspan=4)
	plt.pcolor(-c,cmap=plt.get_cmap('RdYlBu'),vmin=-1,vmax=1)
	plt.plot(I,s,'-w')
	ax.axvline(l[2]-.5,color='g')
	ax.axvline(m,color='m')
	ax.axvline(v,color='r')
	fig.show()


def DMI_plot(t,x,br,S=9,rsp=6,xlim=None):
	n = 20
	s = np.arange(2**S)
	ws = range(5,S+1)
	rows = len(ws) + rsp + 2
	
	fig = plt.figure(figsize=(8,10))
	ax = [plt.subplot2grid((rows,1),(0,0),rowspan=rsp)]
	
	try: 
		d = np.array(mpld.date2num(t))
		br = mpld.date2num(br)
		ax[-1].xaxis.set_major_formatter(mpld.DateFormatter('%Y'))
	except: 
		d = t
		br = np.array(br)-.5
	d2 = d[:-1] + np.diff(d)
	c = CWT(x,s)
		
	def _text(s,ax):
		bb = ax[-1].get_position()
		fig.text(bb.x0+0.01,bb.y1-0.01,s,size=16,weight='bold',va='top')
	
	
	plt.pcolor(d,s,c,cmap=plt.get_cmap('coolwarm'),vmin=-2,vmax=2)
# 	I = trace(c)
# 	plt.plot(d[I],s,'-w',lw=2)
	M = np.array(np.hstack(mtrace(c)),dtype=int)
 	for i in range(M.shape[1]):
		plt.plot(d[M[:,i]],s,'-w',lw=2)
	ax[-1].set_ylim((0,s[-1]))
	ax[-1].invert_yaxis()
	ax[-1].xaxis.tick_top()
	
	for i in ws:
		ax += [plt.subplot2grid((rows,1),(len(ax)+rsp-1,0))]
		plt.plot(d,np.abs(c[2**i-1,:]),color='grey')
		plt.plot(d[M[2**i-1,:]],abs(c[2**i-1,M[2**i-1,:]]),'x',color='r',mew=2)
		_text('{}'.format(2**i),ax)
		ax[0].axhline(2**i-1,color='grey',ls='--')
	
	W,mW = Wilcoxon(x,n)
	T,mT = maxT(x,n)
	ax += [plt.subplot2grid((rows,1),(len(ax)+rsp-1,0))]
	pl, = plt.plot(d2[n:-n],T[n:-n])
	ylim = ax[-1].get_ylim()
	pl.remove()
	plt.plot(d2,T,color='b')
	ax[-1].set_ylim(ylim)
	_text('t',ax)
	
	ax += [plt.subplot2grid((rows,1),(len(ax)+rsp-1,0))]
	pl, = plt.plot(d2[n:-n],W[n:-n])
	ylim = ax[-1].get_ylim()
	pl.remove()
	plt.plot(d2,W,color='b')
	ax[-1].set_ylim(ylim)
	_text('Wilcoxon',ax)
	
	for a in ax[1:]:
		a.set_xticks(())
		a.yaxis.set_major_locator(MaxNLocator(nbins=3))
	
	for a in ax:
		a.set_xlim((d[0],d[-1]))
		for l in a.get_yticklabels():
			l.set_fontsize(10)
		for b in br: 
			a.axvline(b,color='lime',lw=2)
	
	ymax = max([a.get_ylim()[1] for a in ax[1:len(ws)+1]])
	for a in ax[1:len(ws)+1]:
# 		a.axvline(d[m],color='magenta',lw=2)
		a.set_ylim((0,ymax))
		
	ax[-2].axvline(d2[mT],color='red',lw=2)
	ax[-1].axvline(d2[mW],color='red',lw=2)
	
	if xlim!=None:
		for a in ax:
			a.set_xlim(xlim)
	
	fig.show()
	return fig,ax[0].get_xlim(),M

def trace(C):
	ma,mi = np.argmax(C[-1,:]),np.argmin(C[-1,:])
	I,S = ([ma],1) if np.abs(C[-1,ma])>np.abs(C[-1,mi]) else ([mi],-1)
	for j in range(1,C.shape[0]):
		w = C[-1-j,:]
		u = np.abs(w)
		a,b = [set((1+np.where(u[1:-1]-v>=0)[0]).tolist()) for v in (u[:-2],u[2:])]
		c,d = [set((1+np.where(u[1:-1]-v>0)[0]).tolist()) for v in (u[:-2],u[2:])]
		i = [n for n in list((a&b) & (c|d)) if np.sign(2*w[n]-w[n-1]-w[n+1])==S]
		I.insert(0,i[np.argmin(np.abs(I[0]-np.array(i)))])
	return I

def mtrace(C):
	P = np.r_['0,2',sig.argrelmax(C[-1,:])]
	N = np.r_['0,2',sig.argrelmin(C[-1,:])]
	for s in range(1,C.shape[0]):
		p, = sig.argrelmax(C[-1-s,:])
		n, = sig.argrelmin(C[-1-s,:])
		P = np.vstack(([p[np.argmin([np.abs(i-j) for j in p])] for i in P[0,:]],P))
		N = np.vstack(([n[np.argmin([np.abs(i-j) for j in n])] for i in N[0,:]],N))
	return P,N
		
def slope(C):
	m = [i[0] for i in sorted(enumerate(np.abs(np.mean(C,0))),key=lambda x:x[1],reverse=True)[:50]]
	v = [j[0] for j in sorted([(i,np.var(C[:,i])) for i in m],key=lambda x:x[1])]
	return m[0],v[0]

def maxT(x,n=20):
# Wang et al., 2007
	N = len(x)
	T = []
	for k in range(1,N):
		X1,X2 = np.mean(x[:k]),np.mean(x[k:])
		sk = (np.sum((x[:k]-X1)**2) + np.sum((x[k:]-X2)**2)) / (N-2)
		T.append( ( k*(N-k) / (N * sk) ) **.5 * abs(X1-X2) )
	return T,np.argmax(T[n:-n]) + n

def TT(x,n=20):
	from scipy.stats import ttest_ind
	P = []
	print 'TT'
	for c in range(1,len(x)):
		t,p = ttest_ind(x[:c],x[c:])
		P.append(1-p)
	return P,np.argmax(P[n:-n])+n

def Wilcoxon(x,n=20):
# Reeves et al., 2007
	N = len(x)
	r = np.zeros(N)
	for i,j in enumerate(np.argsort(x)[::-1]): r[j] = i
	W = []
	for c in range(1,N):
		n1,n2 = c,(N-c-1)
		u = 12 * ((np.sum(r[:c]) - c*(N+1)/2.)**2) / (c*(N-c)*(N+1))
		W.append( u )
	return W,np.argmax(W[n:-n]) + n

def MWU(x,n=20):
	from scipy.stats import mannwhitneyu
	P = []
	print 'MWU'
	for c in range(1,len(x)):
		u,p = mannwhitneyu(x[:c],x[c:])
		P.append(1-2*p)
	return P,np.argmax(P[n:-n])+n

	


def breaks(id):
	b = []
	for s in Session.query(Station).filter(Station.station_id==id).order_by(Station.startdate):
		if s.enddate: b.append(s.enddate)
	return b


if __name__ == "__main__":
# 	L = data()
# 	l = L[4]
# 	t = np.arange(len(l[1]))
# 	fig = DMI_plot(t,l[1],[l[2]],False)
	
# 	x = sio.loadmat('../AGU2015/surrogate.mat')['surrogate']
# 	i = np.random.randint(0,x.shape[0])
# 	j = np.random.randint(.2*x.shape[1],.8*x.shape[1])
# 	t = np.arange(x.shape[1])
# 	y = np.hstack((x[i,:j],x[i,j:]+2.))
# 	
# 	S = 9
# 	fig,xlim,M = DMI_plot(t,y,[j],S=S)
# 	
# 	s = np.arange(2**S)
# 	k = np.argmin(np.abs(M[0,:]-j))
# 	fig2 = plt.figure()
# 	for l in range(M.shape[1]):
# 		if k==l:
# 			plt.plot(np.abs([c[i,M[i,l]] for i in s]),color='r',lw=2,label='{}'.format(M[0,l]))
# 		else:
# 			plt.plot(np.abs([c[i,M[i,l]] for i in s]),color='grey',label='{}'.format(M[0,l]))
# 	plt.loglog(basex=2,basey=2)
# 	fig2.show()
	
	from schema import *
	f = Session.query(Field).get(2418) # Tassilaq monthly long
	
	g = Session.query(Field).get(2403) # Itto previous
	h = Session.query(Field).get(2408) # Ittoqqortoormiit monthly long
	
# 	f = Session.query(Field).get(2383) # Narsarsuaq monthly long
	
	x = np.array([float(r.x)*float(f.mult) for r in f.records])
	t = [r.t.date() for r in f.records]
	fbreaks = [date(1982,3,15),date(1947,10,15),date(1979,7,15),date(1957,12,15),date(2005,8,15)]
# 	fig1,xlim,M1 = DMI_plot(t,x,fbreaks)
	
	x = np.array([float(r.x)*float(g.mult) for r in g.records]+[float(r.x)*float(h.mult) for r in h.records])
	t = [r.t.date() for r in g.records]+[r.t.date() for r in h.records]
	gbreaks = [date(1980,9,15),date(1949,12,15),date(1957,12,15),date(2005,8,15)]
	
# 	fig2,xlim,M2 = DMI_plot(t,x,gbreaks,xlim=xlim)
	
	