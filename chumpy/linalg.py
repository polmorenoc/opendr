#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""


import numpy as np
import scipy.sparse as sp
from ch import Ch, depends_on, NanDivide
from blender_utils import row, col
from chumpy import ch


def main():
    
    tmp = random.randn(100).reshape((10,10))
    print('chumpy version: ' + str(slogdet(tmp)[1].r))
    print('old version:' + str(np.linalg.slogdet(tmp.r)[1]))

    eps = 1e-10
    diff = np.random.rand(100) * eps
    diff_reshaped = diff.reshape((10,10))
    print(np.linalg.slogdet(tmp.r+diff_reshaped)[1] - np.linalg.slogdet(tmp.r)[1])
    print(slogdet(tmp)[1].dr_wrt(tmp).dot(diff))
    
    print(np.linalg.slogdet(tmp.r)[0])
    print(slogdet(tmp)[0])

if __name__ == '__main__':
    main()

