import numpy as np

def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=1000
    for i in range(iterations):
        y_pred=m_curr*x+b_curr


x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)