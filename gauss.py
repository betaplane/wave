#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import data


# var = 'temp'
# type = 'sur1'
# S = data.benchmark(var,type,0,1)
# t = S.t.astype(date)
# x = S.inho

S = data.DMI(4360)
t = S.t.astype(date)
x = S.x
x[np.isnan(x)] = 0


def d2G(s):
	x = np.arange(-3*s,3*s+1,dtype='float')
	return np.exp(-.5*(x/s)**2) * (x**2-s**2) #/ ((2*np.pi)**.5*s**5)

def d2F(x,s):
	f = np.fft.fftfreq(len(x))
	y = np.fft.fft(x) * np.exp(-.5*(f*s)**2) * f**2 #/ (2*np.pi)**.5
	return np.real(np.fft.ifft(y))


def conv(x,g):
	l = (len(g)-1)/2
	y = np.hstack((x,x[::-1]))
	return np.roll(np.convolve(y,g[::-1],'valid'),l)[:len(x)]	

if __name__ == "__main__":
# 	y = [conv(x,d2G(s)) for s in range(1,200)]
	y = [d2F(x,s) for s in range(5,20000,50)]
	
	fig = plt.figure()
	ax = plt.subplot()
	plt.contour(y,[0])
	S.plot_vlines(ax,False)
	fig.show()