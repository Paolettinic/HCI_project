import numpy as np
def rotationX(alpha):
    return np.array([
        [1, 0,              0],
        [0, np.cos(alpha),  -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)],
    ])

def rotationY(beta):
    return np.array([
        [np.cos(beta),  0,  np.sin(beta)],
        [0,             1,  0],
        [-np.sin(beta), 0,  np.cos(beta)],
    ])

def rotationZ(gamma):
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma),  0],
        [0,             0,              1],
    ])

def getRotationMatrix(alpha,beta,gamma):
    return np.multiply(
        rotationX(alpha),
        rotationY(beta),
        rotationZ(gamma)
    )
