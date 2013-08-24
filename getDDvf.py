from math import cos, sin

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
