import numpy as np
def trim_values(x):
    return x.replace(' ', np.nan)
def to_float(x):
    return x.astype(float)