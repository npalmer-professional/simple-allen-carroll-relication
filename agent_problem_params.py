'''
Set up parameters for agent problem here.
'''
import numpy as np

# Agent params:
rho = 3.0
beta = 0.95

# Income and interest rate:
R       = 1.0
Ypoints = np.array([0.7, 1.0, 1.3]) 
Yprobs  = np.array([0.2, 0.6, 0.2])

# Seeds
sim_seed = 2345642
income_seed = 1234567890

