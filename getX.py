from getDDvf import getDDvf

def getX(im, x, y, thetas, epsilon):
    '''Get gradient for n evenly-spaced radial directions around [x,y]'''
    return [getDDvf(im, x, y, theta, epsilon) for theta in thetas]
