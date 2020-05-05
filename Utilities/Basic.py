import numbers
import collections

import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs

def isCollection(item):
    return isinstance(item, (collections.Sequence, np.ndarray))

def isNumeric(val):
    return isinstance(val, numbers.Number)
