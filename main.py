from __future__ import division, print_function
from builtins import zip, range, object
import numpy as np
import agent_problem_params as param
from basic_consumption_savings import simulation

# ------------------------------------------------------------------------------
# --------------------- Set up agent simulation parameters ---------------------
# ------------------------------------------------------------------------------

I = 50      # Number of agents to run in simulation
M = 5       # Number of trials 
T = 10      # Length of trials
m0 = 1.0    # Initial value of "cash on hand"

# construct rules:
mpc_min, mpc_max, mpc_N     = 0.0, 1.0, 10
mbar_min, mbar_max, mbar_N  = 1.0, 3.0, 10

rules = []
for mpc in np.linspace(mpc_min, mpc_max, mpc_N):
    for mbar in np.linspace(mbar_min, mbar_max, mbar_N):
        rules.append( (mpc, mbar) )

# Append the best rule:
rules.append( (0.25, 1.2))

# ------------------------------------------------------------------------------
# --------------------- Set up and run simulation ------------------------------
# ------------------------------------------------------------------------------

sim = simulation(beta=param.beta, rho=param.rho, R=param.R,
                 I=I, T=T, M=M, rules=rules,
                 m0=m0, Ey0=None,
                 income_values=param.Ypoints,
                 income_probs=param.Yprobs,
                 income_seed=param.sim_seed)

sim.run_simulation()

result = sim.calculate_sacrifice_fractions()



