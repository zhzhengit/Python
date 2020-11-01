#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
class Perceptron(object):
    
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0] * input_num
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n'% (self.weights, self.bias)

    def sum_h(self,x):
        return self.activator(self.weights[0]*x[0]+self.weights[1]*x[1]+self.bias)
    
    def train(self,x,t,iteration,rate):
        for i in range(iteration):
            self.one_iteration(x,t,rate)
            
    def one_iteration(self,x,t,rate):
        for x,t in zip(x,t):
            output = self.sum_h(x)
            self.update_weights(x,t,output,rate)
            
    def update_weights(self,x,t,output,rate):
        dalta = t -output
        self.weights[0] = self.weights[0] + rate*dalta*x[0]
        self.weights[1] = self.weights[1] + rate*dalta*x[1]
        self.bias = self.bias + rate*dalta
    
def f(x):
    if x>0:
        return 1
    else: 
        return 0
def get_data():
    x_train = np.array([[0,0],[1,0],[0,1],[1,1]])
    y_train = np.array([0,0,0,1])
    return x_train,y_train

def train_and_perceptron():
    p = Perceptron(2,f)
    x,t = get_data()
    p.train(x,t,10,0.1)
    return p
if __name__=='__main__':
    a = train_and_perceptron()
    print(a)
    print('1 and 1 = %d' % a.sum_h([1, 1]))
    print('0 and 0 = %d' % a.sum_h([0, 0]))
    print('1 and 0 = %d' % a.sum_h([1, 0]))
    print('0 and 1 = %d' % a.sum_h([0, 1]))


# In[ ]:




