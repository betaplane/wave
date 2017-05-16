#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from matplotlib.figure import SubplotParams


def CWT(x,J):
	N = len(x)
	S = np.fft.fft(np.hstack((x,x[::-1])))
	f = 2 * np.pi * np.fft.fftfreq(2*N)
	D = np.zeros((J,N))
	for j in range(J):
		w = (j+1) * f
		W = 1j*w * np.sinc(w/(4*np.pi))**4
		D[j,:] = np.real(np.fft.ifft(S * W)[:N])
	return D

def ll(fig):
	ax = fig.axes[0]
	if ax.get_xscale()=='linear':
		ax.set_xscale('log')
		ax.set_yscale('log')
	else:
		ax.set_xscale('linear')
		ax.set_yscale('linear')
	ax.figure.canvas.draw()

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


L,s = 1000,300
g = gaussian(50,5)
g = g/np.sum(g)
x1 = np.hstack((np.zeros(L/2),np.ones(L/2)))
x2 = np.convolve(np.hstack((np.zeros(L/2),np.ones(L))),g,'same')[:L]

g = gaussian(200,20)
g = g/np.sum(g)
x3 = np.convolve(np.hstack((np.zeros(L/2),np.ones(L))),g,'same')[:L]
W1 = CWT(x1,s)
W2 = CWT(x2,s)
W3 = CWT(x3,s)


fig1=plt.figure(figsize=(6,6),subplotpars=SubplotParams(left=.1))
ax = plt.subplot2grid((3,1),(0,0),rowspan=2)
plt.pcolor(W1,cmap=plt.get_cmap('viridis'))
plt.plot(s/3*W1[100]+100,'-w',lw=2)
ax.axvline(L/2,color='magenta',lw=2)
ax.set_xticklabels([])
ax.set_ylabel('scale')
ax = plt.subplot2grid((3,1),(2,0))
plt.plot(x1,'-r',lw=2)
ax.set_ylim((-.5,1.5))
fig1.show()


# fig2=plt.figure()
# ax = plt.subplot2grid((3,1),(0,0),rowspan=2)
# plt.plot(W1[:,L/2],'-b',lw=2)
# plt.plot(W2[:,L/2],'-g')
# plt.plot(W3[:,L/2],'-r')
# x = np.arange(501)
# plt.plot(x,x/10.,'-k',lw=2)
# ax.text(2,.15,r"$\alpha=1$")
# plt.loglog()
# ax.set_xlim((1,s))
# ax.set_ylim((.1,1.7))
# ax.set_xlabel('scale')
# ax = plt.subplot2grid((3,1),(2,0),rowspan=1)
# plt.plot(x2,'-g')
# plt.plot(x3,'-r')
# ax.set_ylim((-.5,1.5))
# fig2.show()



# x = np.hstack((np.random.normal(0,1,500),np.random.normal(.5,1,500)))
# W = CWT(x,500)
# 
# fig=plt.figure()
# plt.pcolor(W,cmap=plt.get_cmap('viridis'),vmin=-.5,vmax=.5)
# fig.show()

x = np.zeros(L)
x[L/2] = 1
W = CWT(x,s)


I = trace(W)


fig3=plt.figure(figsize=(6,6),subplotpars=SubplotParams(left=.1))
ax = plt.subplot2grid((3,1),(0,0),rowspan=2)
plt.pcolor(W,cmap=plt.get_cmap('coolwarm'),vmin=-.3,vmax=.3)
plt.plot(s*W[100,:]+100,'-k',lw=2)
plt.plot(I,range(len(I)),color='magenta',lw=2)
ax.set_xticklabels([])
ax.set_ylabel('scale')
ax.set_ylim((0,s))
# ax.set_yticks((0,100))
ax = plt.subplot2grid((3,1),(2,0))
plt.plot(x,lw=2)
ax.set_ylim((-.5,1.5))
fig3.show()


fig4=plt.figure(figsize=(6,6),subplotpars=SubplotParams(left=.1))
ax = plt.subplot()
plt.plot(W1[:,L/2],'-r',lw=2)
plt.plot(np.abs([W[i,j] for i,j in enumerate(I)]),'-b',lw=2)
x = np.arange(501)
plt.plot(x,1./x,'-k',lw=2)
# ax.text(2,.15,r"$\alpha=-1$",fontsize=16,weight='bold')
plt.loglog()
ax.set_xlim((1,s))
ax.set_ylim((.01,1.7))
ax.set_xlabel('scale')
for a in ax.spines.values():
	a.set_color('magenta')
fig4.show()

fig1.savefig('../AGU2015/step_wavelet.png',dpi=600)
# fig2.savefig('../AGU2015/step_smoothed.eps',dpi=600)
fig3.savefig('../AGU2015/dirac_wavelet.png',dpi=600)
fig4.savefig('../AGU2015/slopes.eps',dpi=600)

