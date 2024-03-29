import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s),(s,c)))
    return R