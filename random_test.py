#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import random

x = [random.gauss(0,1) for i in range(12)]
x.extend(x)

y = np.convolve(x,np.ones(6)/6,'valid')
y = np.convolve(y,np.ones(6)/6,'valid')
fig=plt.figure()
plt.plot(y)
fig.show()