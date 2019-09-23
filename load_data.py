import numpy as np

def load_data(filename):
    '''
    General case to load any data from filename
    Output tuple:
        xtr, ytr : Data and labels for training
        xva, yva : Data and labels for validation
        xte, yte : Data and labels for test
    '''
    loaded = np.load(filename)
    xtr = loaded['xtr']
    ytr = loaded['ytr']
    xva = loaded['xva']
    yva = loaded['yva']
    xte = loaded['xte']
    yte = loaded['yte']
    return (xtr, ytr, xva, yva, xte, yte)
    