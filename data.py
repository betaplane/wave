#!/usr/bin/env python
import csv, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from schema import Station, Session
import matrices as mat


class Break(object):
	def __init__(self,parent,type,year,month):
		self.date = date(year,month,15)
		self.index = np.where(parent.t==np.datetime64('{}-{:02d}'.format(year,month)))[0][0]
		if type=='OUTLIE':
			self.type = 'out'
		else:
			self.type = 'break'
			x = parent.x - parent.homo
			self.jump = np.mean(x[self.index+1]) - np.mean(x[self.index])
		parent.breaks.append(self)
			
		

class benchmark(object):
	def __init__(self,var,type,*j):
		self.breaks = []
		self.outliers = []
		t = []
		net = os.listdir(os.path.join('benchmark','inho',var,type))[j[0]]
		for h in ('inho','orig'):
			d = []
			path = os.path.join('benchmark',h,var,type,net)
			fname = [f for f in os.listdir(path) if f[0]!='0'][j[1]]
			with open(os.path.join(path,fname)) as file:
				if h=='inho':
					for r in csv.reader(file,delimiter='\t'):
						d.extend([float(x) for x in r[1:13]])
						t.extend(['{}-{:02d}'.format(r[0],m) for m in range(1,13)])
					self.x = np.array(d)
					self.t = np.array(t,dtype='datetime64[M]')
				else:
					for r in csv.reader(file,delimiter='\t'):
						d.extend([float(x) for x in r[1:13]])
					self.homo = np.array(d)
		self.missing()
					
		with open(os.path.join(path,'{}detected.txt'.format(net))) as file:
			for r in csv.reader(file,delimiter='\t'):
				if r[0]==fname:
					b = Break(self,r[1],int(r[2]),int(r[3]))		
	
	def missing(self):
		i = np.where(self.x!=-999.9)[0]
		j = np.arange(i[0],i[-1]+1)
		i = np.where(self.x==-999.9)[0]
		self.x[i] = float('nan')
		self.homo[i] = float('nan')
		for a in ('x','homo','t'):
			setattr(self,a,getattr(self,a)[j])
			
	
	def plot_vlines(self,ax,date=True):
		for b in self.breaks:
			if b.type=='break':
				c = 'r' if b.jump>0 else 'b'
				if date: ax.axvline(b.date,color=c)
				else: ax.axvline(b.index+.5,color=c)

				
class DMI(object):
	def __init__(self,id):
		R = mat.Session.query(mat.Matrix).get(4).dbmatrix
		self.t = R.cols
		self.x = R[np.where(R.rows==id)[0][0],:]
		self.breaks = []
		for s in Session.query(Station).filter(Station.station_id==id).order_by(Station.startdate):
			try: self.breaks.append((s.startdate,s.enddate))
			except: self.breaks.append((s.startdate,))

	def plot_vlines(self,ax,date=True):
		if date:
			for b in self.breaks:
				ax.axvline(b[0],color='g')
				try: ax.axvline(b[1],color='r')
				except: pass
		else:
			for b in np.array(self.breaks,dtype=self.t.dtype):
				l = np.where(self.t==b[0])[0][0]
				ax.axvline(np.where(self.t==b[0])[0][0],color='g')
				try: ax.axvline(np.where(self.t==b[1])[0][0],color='r')
				except: pass
	