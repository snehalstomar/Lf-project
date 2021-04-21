import numpy as np

def LFRotz(psi):
    c = np.cos(psi)
    s = np.sin(psi)

    R = np.array([[c, -s,  0],
                  [s,  c,  0],
                  [0,  0,  1]])
    
    return R