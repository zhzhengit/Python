#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import math
from matplotlib import pyplot as plt
y = []
x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
y.append(1)
t = np.linspace(0,1,100)
def gra(x,y):
    return  y-(2*x)/y
def gradient():
    for i in range (0,10):
        y_p = y[i] + 0.1*gra(x[i],y[i])
        y_c = y[i] + 0.1*gra(x[i+1],y_p)
        y.append(1/2*(y_p + y_c))
        plt.scatter(x[i],y[i], c='b', s=50, alpha=0.5)
    y1 = pow((1+2*t),1/2)
    plt.plot(t,y1,'r')
gradient()


# In[ ]:




