import numpy as np

def taylor_green_vortex(x, y, t, nu):
    u = -np.cos(x)*np.sin(y)*np.exp(-2*nu*t)
    v = np.sin(x)*np.cos(y)*np.exp(-2*nu*t)
    return u, v

def linear_advection(x, y, t, c=1.0):
    return np.sin(x - c*t) * np.cos(y - c*t) 