from math import sin, cos
import numpy as np
import Image
import ImageDraw

from getX import getX
from h_ax import h_ax

class Agent:
    def __init__(self, alpha):
        self.alpha = alpha
        self.path = None
        self.finished = None
    
    def render_path(self, im):
        path_img = im.copy()
        draw = ImageDraw.Draw(path_img)
        draw.line(self.path, fill=(255,0,0))
        return path_img
    
    def score(self, im, x, y, epsilon, lam, n, maxN, Q):
        '''Compute the score of this agent on the provided image.'''
        thetas = np.linspace(0, 2*np.pi, n+1)[:-1]
        self.path = [(x,y)]
        cost = 0.0
        self.finished = False
        for i in range(maxN):
            # Compute model output 
            X = getX(im, x, y, thetas, epsilon)
            phi_hat = h_ax(self.alpha, X)
            
            # Compute next position based on model output
            x_next = round(x + epsilon*cos(phi_hat))
            y_next = round(y + epsilon*sin(phi_hat))

            # Clamp next position to the image boundary
            x_next = np.clip(x_next, 0, im.shape[1] - 1)
            y_next = np.clip(y_next, 0, im.shape[0] - 1)

            # Update the cost
            a = im[y, x]
            b = im[y_next, x_next]
            delta_n = float(b - a)
            if delta_n >= 0:
                g_n = delta_n
            else:
                g_n = - delta_n /  2
            cost += g_n**2
            
            self.path.append((x_next, y_next))
            
            # Stop if we've reached the goal
            if x_next == im.shape[1] - 1:
                self.finished = True
                break
                
            x, y = x_next, y_next
        
        # Add the penalty if the goal wasn't reached
        if not self.finished:
            cost += Q
        
        cost += lam * len(self.path)
        
        return cost
