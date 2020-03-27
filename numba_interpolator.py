'''
Goal: implement a numba-friendly 1d interpolator. 


Will use this tutorial:  

- http://numba.pydata.org/numba-doc/latest/user/jitclass.html#jitclass
- full code example:
    - https://github.com/numba/numba/blob/master/examples/jitclass.py

...to get friendly linear spline working. 

Note that:
    numpy.interp
    numpy.interp(x, xp, fp, left=None, right=None, period=None)[source]



'''


"""
A simple jitclass example.
"""

import numpy as np
from numba import jitclass                  # import the decorator
from numba import int32, float32, float64   # import the types

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]


@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] += val
        return self.array

mybag = Bag(21)
print('isinstance(mybag, Bag)', isinstance(mybag, Bag))
print('mybag.value', mybag.value)
print('mybag.array', mybag.array)
print('mybag.size', mybag.size)
print('mybag.increment(3)', mybag.increment(3))
print('mybag.increment(6)', mybag.increment(6))
        

# Now let's replicate this with numba-ified interpolator: 

spec = [
    ('xp', float32[:]),          # a simple scalar field
    ('yp', float32[:]),          # an array field
]


@jitclass(spec)
class Interp(object):
    def __init__(self, xp, yp):
        self.xp = xp
        self.yp = yp

    def __call__(self, x):
        return np.interp(x, self.xp, self.yp)


xtest = np.array([1.0, 2.0, 4.25, 3.0])
ytest = np.array([1.0, 2.0, 4.25, 3.0])

f = Interp(xtest, ytest)
print("f(1.5):",f(1.5))
print("f(0.5):",f(0.5))
print("f(5.5):",f(5.5))
        
        
        
        
        
        
        
