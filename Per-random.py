#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
class Perceptron(object):
    
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [10.0] * input_num
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n'% (self.weights, self.bias)

    def sum_h(self,x):
        return self.activator(self.weights[0]*x[0]+self.bias)
    
    def train(self,x,t,iteration,rate):
        for i in range(iteration):
            self.one_iteration(x,t,rate)
            
    def one_iteration(self,x,t,rate):
        for x,t in zip(x,t):
            output = self.sum_h(x)
            self.update_weights(x,t,output,rate)
            
    def update_weights(self,x,t,output,rate):
        dalta = t-output
        self.weights[0] = self.weights[0] + rate*dalta*x[0]
        self.bias = self.bias + rate*dalta
    
def f(x):
    return x
def get_data():
    x_train = np.array([[5],[3],[8],[1.4],[10.1]])
    y_train = np.array([5500,2300,7600,1800,11400])
    return x_train,y_train

def train_linear_unit():

    lu = Perceptron(1,f)
    x,t = get_data()
    lu.train(x,t,100,0.01)
    return lu

def plot(a):
    import matplotlib.pyplot as plt
    xx, yy = get_data()
    fig = plt.figure()
    weights = a.weights
    bias = a.bias
    x =np.linspace(1,10,100)
    y =weights*x+bias
    plt.plot(x, y,'r')
    plt.grid()
    plt.scatter(xx,yy)
    plt.show()

if __name__=='__main__':
    a = train_linear_unit()
    print(a)
    plot(a)
    print ('Work 3.4 years, monthly salary = %.2f' % a.sum_h([3.4]))


# In[ ]:




