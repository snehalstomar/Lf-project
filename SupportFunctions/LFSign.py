import numpy as np

def LFSign(i):
    s = np.ones(np.shape(i))
    s[i<0] = -1
    return s