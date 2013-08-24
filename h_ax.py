from math import pi, exp
import numpy as np

def h_ax(alpha, x):
    '''h_alpha(x) = 2*pi(sigmoid(x)-0.5)'''
    return 2*pi*(1.0 / (1+exp(np.dot(alpha, x))) - 0.5)
