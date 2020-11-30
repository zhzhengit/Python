#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from random import *

class neural(object):
    def __init__(self,activator):
        self.activator = activator
        self.W1 = np.random.randn(4,3)     
        self.W2 = np.random.randn(2,4)   
        self.b1 = np.random.randn(1, 4)                 
        self.b2 = np.random.randn(1, 2) 

    def __str__(self):
        print('W1:',self.W1,'W2:',self.W2,'b1:',self.b1,'b2:',self.b2)
    
    def sum_h(self,x):
        a=[]
        for i in range(4):
        
            s=self.W1[i][0]*x[0]+self.W1[i][1]*x[1]+self.W1[i][2]*x[2]+self.b1[0][i]
            s=self.activator(s)
            a.append(s)   
        return a
        
    
    def train(self,x,t,iteration,rate):
        for i in range(iteration):
            self.one_iteration(x,t,rate)
            
    def one_iteration(self,x,t,rate):
        for x,t in zip(x,t):
            self.updata_weights(x,t,rate)
    def qianxiangjisuan(self,x,t):
        a = self.sum_h(x)
        b = self.activator(a[0]*self.W2[0][0] +a[1]*self.W2[0][1]+a[2]*self.W2[0][2]+a[3]*self.W2[0][3]+self.b2[0][0])
        c = self.activator(a[0]*self.W2[1][0] +a[1]*self.W2[1][1]+a[2]*self.W2[1][2]+a[3]*self.W2[1][3]+self.b2[0][1])
        return b,c
    def houxiangjisuan(self,x,t):
        a = self.sum_h(x)
        y1,y2 = self.qianxiangjisuan(x,t)
        loss1=[]
        loss2=[]
        ##计算输出层误差
        loss1.append(y1*(1-y1)*(t[0]-y1))
        loss1.append(y2*(1-y2)*(t[1]-y2))
        ##计算隐藏层误差
        for i in range(4):
            s = (a[i]*(1-a[i])*(self.W2[0][i]*loss1[0]+self.W2[1][i]*loss1[1]))
            loss2.append(s)
        return loss1,loss2
    
    def updata_weights(self,x,t,learning):
        ##更新W1
        loss1,loss2=self.houxiangjisuan(x,t)
        for i in range(4):
            for j in range(3):
                self.W1[i][j]+= learning*loss2[j]         
        ##更新W2    
        for i1 in range(2):
            for j1 in range(4):
                self.W2[i1][j1]+= learning*loss1[i1]
        ##更新偏置项
        for i2 in range(4):
            self.b1[0][i2]+= learning * loss2[i2]
        for i3 in range(2):
            self.b2[0][i3]+= learning * loss1[i3]
        
        
        
def f(x):
    return 1.0 / (1 + np.exp(-x))

def get_data():
    X = np.array([[3,1,100],  [400,1,120],  [3,50,100],  [4,50,10]])    
    t = np.array([[0,1],[1,0],[1,1],[1,0]]) 
    np.random.seed(1)
    return X,t

def train_unit():
    x,t = get_data()
    lu = neural(f)
    lu.train(x,t,100,0.05)
    return lu

if __name__=='__main__':
    a = train_unit()
    a.__str__()


# In[ ]:




