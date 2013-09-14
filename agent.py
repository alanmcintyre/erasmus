from math import sin, cos, pi, exp
import numpy as np
import Image
import ImageDraw

def getDDvf(im, x, y, theta, epsilon):
    '''Approximate (w/ epsilon) the directional derivative of f at (x,y)
    along a vector defined by theta.'''
    x2 = x + epsilon*cos(theta)
    y2 = y + epsilon*sin(theta)

    x = max(x,0)
    y = max(y,0)

    x2 = max(x2, 0)
    y2 = max(y2, 0)

    x2 = min(x2, im.shape[1]-1)
    y2 = min(y2, im.shape[0]-1)

    x2 = round(x2)
    y2 = round(y2)

    p1 = im[y, x]
    p2 = im[y2, x2]
    return (p2-p1)/epsilon

def getX(im, x, y, thetas, epsilon):
    '''Get gradient for n evenly-spaced radial directions around [x,y]'''
    return [getDDvf(im, x, y, theta, epsilon) for theta in thetas]

def h_ax(alpha, x):
    '''h_alpha(x) = 2*pi(sigmoid(x)-0.5)'''
    return 2*pi*(1.0 / (1+exp(np.dot(alpha, x))) - 0.5)

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
