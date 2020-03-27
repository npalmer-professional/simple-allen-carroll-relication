'''
1. The *basic* consumption-savings agents
    a. choices: c
    b. states:  m
    c. shocks:  y;     prices:  R

'''
from __future__ import division, print_function
from builtins import zip, range, object
import pickle
import numpy as np
from scipy import stats
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from scipy.stats.mstats import mquantiles
from scipy.interpolate import InterpolatedUnivariateSpline


class learning_agent(object):

    def __init__(self, beta, rho, Ey0):
        # Define the agent -- set up agent with required attributes, some of 
        # which will be filled with placeholders. 
        # beta: discount factor
        # rho: crra preference parameter
        # Ey0: initial expected income
        
        # Prefernce parameters: discount factor & risk aversion, respectively
        self.beta, self.rho = beta, rho

        # Define m: total cash-on-hand, includes assets from last period and 
        # current-period income. Fill with plaeholder. 
        self.m = None
        
        # Define consumption function parameter placeholders
        self.mpc  = None    # marginal propensity to consume; "gamma" in A&C2001
        self.mbar = None    # buffer stock savings target; "X-bar" in A&C 2001
        self.theta = (self.mpc, self.mbar)  # Following paper, theta is both
        self.Ey = Ey0   # Note: E[y] enteres the consumption function but is not
                        # a parameter the agent chooses in the baseline model. 
                        # This will be provided by the modeler to reflect the 
                        # true nature of the world. There is a straightforward 
                        # way to endogenize this in future versions. 

        # Initilize value estimate to placeholder
        self.w = None
        
        # Define a place-holder for the compounded beta; keeps track of beta^t
        self.beta_compounded = None

        # Define a utility function; address the special case of rho == 1.0:
        if self.rho == 1.0:
            self.u = np.log
        else:
            self.u = lambda c: c**(1.0 - self.rho) / (1.0 - self.rho) 
        
        # Define a container to hold all estimates of value: 
        self.memory = {} # will index this by key-value pairs. Note: be cautious 
        # about float inaccuracies in keys. 


    def initialize_agent_for_simulation(self, mpc0, mbar0, m0):
        # Fill in placeholders that are required for actually simulating.
        # May be used to re-init an agent object (instead of re-creating) if no
        # other parameters (preferences, Ey) are changed.
        # mpc0: new marginal propensity to consume
        # mbar0: new buffer stock savings target
        # m0: initial cash-on-hand
        
        # Update consumption function values:
        self.update_theta(mpc0, mbar0)
        
        # Update cash-on-hand m: 
        self.m = m0
        
        # Re-initilize value estimate to placeholder
        self.w = None
        
        # Re-initilize compounded beta to a place-holder; keeps track of beta^t
        self.beta_compounded = None

    def update_theta(self, mpc_new, mbar_new):
        # Update the two main parameters of the consumption function.
        # Note this needs to be done before any agent "steps" are taken.
        # mpc_new: new marginal propensity to consume
        # mbar_new: new buffer stock savings target
        self.mpc, self.mbar = mpc_new, mbar_new
        self.theta = (self.mpc, self.mbar)
        
    def c(self, m):
        # Consumption function. 
        # m: cash-on-hand for current period

        return np.maximum(0.0, self.Ey + self.mpc * (m - self.mbar) )

    def remember_rule_result(self):
        # Commit the current w-value to memory for current rule theta
        if self.theta not in self.memory:
            self.memory[self.theta] = [self.w]
        else:
            self.memory[self.theta].append(self.w)
            
    def get_mean_w_from_memory(self, theta):
        return np.nanmean(self.memory[theta])

    def find_best_rule_in_memory(self):
        # Iterate over all rules in memory and return the best
        Theta = self.memory.keys()
        
        max_val = -np.inf
        max_key = None
        for theta in Theta:
            wbar = self.get_mean_w_from_memory(theta)
            if wbar > max_val:
                max_val = wbar
                max_key = deepcopy(theta)

        return max_key, max_val

    def step(self, y, R):
        # Step forward, given:
        # y: current-period income shock
        # R: current-period return on previous-period implied savings
        # Do the following things: 
        # - find previous-period consumption c
        # - update value estimate w with u(c)
        # - update m to current-period using y, c, and R
        
        # Get previous-period consumption from previous-period state m
        c = self.c(self.m)
        
        # Update beta_compounded. If it is None then set it equal to beta:
        if self.beta_compounded is None:
            # Then we were "starting new" for this estimate of w and need to set 
            # beta_compounded to just 1.0 (first-period discounting is none)
            self.beta_compounded = 1.0
        else: 
            # Otherwise update beta_compounded:
            self.beta_compounded *= self.beta 
            
        # Update the current-period value estimate:
        if self.w is None:
            # Then we were "starting new" and set self.w to just "plain" u(c):
            self.w = self.u(c)
        else:
            # Value estimate is updated by compounded discount factor
            self.w += self.beta_compounded * self.u(c)
            
        # Update cash-on-hand m for the current period, using current-period R
        # and income y.
        self.m = (self.m - c) * R + y
        # Done



class simple_income_process(object):

    def __init__(self, values=(0.7, 1.0, 1.3), probs=(0.2, 0.6, 0.2),seed=None):
        # A very simple wrapper around SciPy's discrete_rv 
        # For some reason rv_discrete only allows points to be integers, which
        # is very strange to me. So we will save the *values* and construct a
        # discrete_rv using the *index* to those values. Note of course that 
        # this almost certainly means that we can't use many of the built-in
        # functions *except* drawing random variables. Fortunately thats what 
        # we want to do. 
        # Default values are those of A&C2001:
        #    values=(0.7, 1.0, 1.3), probs=(0.2, 0.6, 0.2)
        # for convenience.

        self.values, self.probs = np.array(values), np.array(probs)
        self.index = np.arange(len(values))
        self.distribution = stats.rv_discrete(name='custm', 
                                              values=(self.index, self.probs),
                                              seed=seed)

    def expect(self, f=lambda x:x):
        x = f(self.values)
        return np.dot(x, self.probs)

    def draw(self, size):
        # Draw random variables of size "size"
        idx = self.distribution.rvs(size=size)
        return self.values[idx]


class simulation(object):

    def __init__(self, beta, rho, R, I, T, M, rules, m0, Ey0=None,
                 income_values=(0.7, 1.0, 1.3), 
                 income_probs=(0.2, 0.6, 0.2), 
                 income_seed=None):
        # Initilize simulation, set up agents
        # beta: discount factor for all agents
        # rho:  risk aversion, all agents
        # R:    risk-free rate of return faced by all agents
        # I:    number of agents
        # T:    Number of periods to use a rule for each trial ("n" in Table 1)
        # M:    Number of trials to use a rule
        # rules: list of tuples of "theta" rules to explore
        # m0:   Starting value of agent cash-on-hand
        # Ey0:  expected income, all agents
        # income_values: discrete values to create income distibution
        # income_probs:  discrete probabilities to create income distribution 
        # income_seed:   seed to generate income draws

        # Save important values:
        self.R = R
        self.T, self.M = T, M

        # Create income generator:
        self.income_rng = simple_income_process(values=income_values, 
                                                probs=income_probs,
                                                seed=income_seed)

        self.Ey0 = Ey0 if Ey0 is not None else self.income_rng.expect() 
        self.m0 = m0
        
        # Create number of agents:
        first_agent = learning_agent(beta=beta, rho=rho, Ey0=self.Ey0)
        self.agents = [first_agent]

        # Fill out list with rest of agents:
        for i in range(1,I):
            self.agents.append(deepcopy(first_agent))

        assert len(self.agents) == I, "Wrong number of agents in list!"
        
        # Save rules to explore: 
        self.rules = rules
        
        # Save container to hold all sacrifice values at the end:
        self.all_sacrifice_values = []
        
        # Save a dictionary of all parameters, just to have in one place:
        self.params = {}
        self.params['beta'], self.params['rho'] = beta, rho
        self.params['Ey0'], self.params['m0'], self.params['R'] = Ey0, m0, R
        self.params['I'], self.params['T'], self.params['M']    = I, T, M
        self.params['rules'] = rules
        self.params['income_values'] = income_values
        self.params['income_probs'] = income_probs
        self.params['income_seed'] = income_seed


    def run_simulation(self):
        # For each rule, have each agent use that rule for the correct number
        # of trials, and the correct number of periods per trial

        self.w_results = {}
        self.rule_results = {}
        for theta in self.rules:
            self.w_results[theta] = []
            self.rule_results[theta] = []

            for agent in self.agents:
                # init agent to current rule and then run M trials
                agent.initialize_agent_for_simulation(mpc0=theta[0], 
                                                      mbar0=theta[1], 
                                                      m0=self.m0)
                for m in range(self.M):
                    # Generate income draw for agent; note that we take total
                    # number of periods as T-1 because the initial income draw
                    # is embeded in m0:
                    y_vec = self.income_rng.draw(size=self.T-1)
                    for y in y_vec:
                        agent.step(y, self.R)
                    # Tell agent to remember rule 
                    agent.remember_rule_result()
            
        # Done at end of this. Final step is analyzing     

    def calculate_sacrifice_fractions(self, sacrifice_pickle="epsbar_mpc_mhats_XYZV.pickle", sacrifice_success_cutoff=0.05):
        # Calculate the number of agents that fall within 
        # sacrifice_pickle: string; pickle file name constructed by "create_simple_sacrifice_value_contour_plot.py"
        # sacrifice_success_cutoff: float; sacrifice value such that agents whose choice 
        #   falls under this threshold
        
        # First load the sacrifice values function
        with open(sacrifice_pickle, 'rb') as a_file:
            mydict = pickle.load(a_file)
        sacrifice_value_function = mydict['interpolated_surface_function']

        # Now loop over all agents and get sacrifice values:
        best_rules = []
        best_values = []
        self.all_sacrifice_values = []
        for agent in self.agents:
            best_theta, best_w = agent.find_best_rule_in_memory()

            best_rules.append(best_theta)
            best_values.append(best_w)
            
            eps_bar = sacrifice_value_function(best_theta)
            if eps_bar < 0:
                print("Warning: eps_bar < 0:",eps_bar, "Setting eps_bar = 1e5")
                eps_bar = 1e5
            self.all_sacrifice_values.append(eps_bar)

        self.all_sacrifice_values = np.array(self.all_sacrifice_values)

        # return the fraction <= sacrifice_success_cutoff
        return np.mean(self.all_sacrifice_values <= sacrifice_success_cutoff)

# ------------------------------------------------------------------------------
# ------ Define optimization "helper" functions for welfare analysis -----------
# ------------------------------------------------------------------------------


def gpi_policy_improvement(v, u, f, beta, grid, tol, Ypoints, Yprobs):
    """
    Obtain a greedy policy given a value function. See Sutton and Barto 1999.

    # Parameters:
    * v: value function
    * u: utility function
    * f: law of motion for state
    * beta: discount factor
    * grid: array of grid points
    * tol:  stopping tolerance
    * Ypoints: array of discrete probability points
    * Yprobs: probs for above points. Should sum to 1.

    Nathan M. Palmer
    June 2013
    """
    grid_min = np.min(grid)
    c_prime = np.empty_like(grid)
    for i, x in enumerate(grid):

        def H(c):
            vprime  = lambda z: v(f(x,c,z))
            return -1*(u(c) + beta*np.dot(vprime(Ypoints), Yprobs))

        c_prime[i] = fminbound(H, grid_min, x)

    return(InterpolatedUnivariateSpline(x=grid, y=c_prime, k=1))


def gpi_value_estimation(v, policy, u, f, beta, grid, tol, Ypoints, Yprobs):
    """
    Iterate to a fixed point value function, given a value function v and
    a policy.  See Sutton and Barto 1999.

    # Parameters:
    * v: value function
    * policy: policy (action) function
    * u: utility function
    * f: law of motion for state
    * beta: discount factor
    * grid: array of grid points
    * tol:  stopping tolerance
    * Ypoints: array of discrete probability points
    * Yprobs: probs for above points. Should sum to 1.

    Nathan M. Palmer
    June 2013
    """

    error = tol + 1
    new_v = np.empty_like(grid)
    while error > tol:
        for i, x in enumerate(grid):
            c = policy(x)
            vprime  = lambda z: v(f(x,c,z))
            new_v[i] = u(c) + beta* np.dot(vprime(Ypoints), Yprobs)
        error = max(np.abs(new_v - v(grid))) #max(np.abs(new_v[1:] - v(grid[1:])))
            # Ignore the first element. Errors often due to CRRA utility
            # function, which is unbounded below/above (depending on r.a.).

        v = InterpolatedUnivariateSpline(x=grid, y=new_v, k=1)

    return(v)


def generalized_policy_iteration(u, f, beta, grid, tol, Ypoints, Yprobs, v=None,
                                 policy=None, verbose=False):
    """
    This is the generalized policy function iteration code I created from
    the Sutton and Barto textbook. 

    Nathan M. Palmer
    June 2013
    """

    if not(v):
        v = InterpolatedUnivariateSpline(x=grid, y=u(grid), k=1)

    if not(policy):
        policy = InterpolatedUnivariateSpline(x=grid, y=grid, k=1)
            #45 deg line.

    first_time = True
    delta = tol*2+1

    while delta > tol:

        if verbose:
            plt.plot(grid, v(grid), 'bo-')
            print("tol = ", tol, "delta = ", delta)

        # Given a value function and a policy function, improve the policy:
        policy = gpi_policy_improvement(v, u, f, beta, grid, tol, Ypoints, Yprobs)

        # Estimate the value function:
        v_prime = gpi_value_estimation(v, policy, u, f, beta, grid, tol, Ypoints, Yprobs)

        # Examine the difference between old value, new value:
        delta = np.max(np.abs( v_prime(grid) - v(grid) ))

        # Update value function:
        v = deepcopy(v_prime)

    if verbose:
        plt.show()

        plt.plot(grid, v(grid), 'bo-')
        plt.show()

        plt.plot(grid, policy(grid), 'ro-')
        plt.show()

    return(policy, v_prime)


def generate_simulated_choice_data(cons, income_generator, periods, Nagents, R, discard_periods=20, m0=None):
    """
    Nathan M. Palmer
    """

    y_shocks = income_generator.draw((periods, Nagents))

    sim_m = np.zeros_like(y_shocks) + np.nan   # Set containers for fast
    sim_c = np.zeros_like(y_shocks) + np.nan   # calculation.

    if m0 is None:
        sim_m[0,] = y_shocks[0,]
    else:
        sim_m[0,] = m0

    sim_c[0,] = cons(sim_m[0,])    # Initilize consumption choice.

    # Generate m-choices:
    for t in range(0,periods-1):

        sim_m[t+1,] = R*(sim_m[t,] - sim_c[t,]) + y_shocks[t+1,]  # record next-period m
        sim_c[t+1,] = cons(sim_m[t+1,])                           # record next-period c

    # Throw out the first discard_periods:
    m = sim_m[discard_periods:,].ravel()
    return (sim_m, sim_c, y_shocks, m)
    
def find_empirical_equiprobable_bins_midpoints(N, data):
    '''
    N number of equiprobable bins and data, return the cutoffs and conditional
    expectation nodes (midpoints), and empirical probability of each bin.
    NOTE that the empirical probability will likely *not* be equal, due to the
    nature of the empirical data. As N_data -> infty, this will converge to the
    appropriate "true" equiprobable discrete value due to properties of the
    ECDF.
    
    Nathan M. Palmer
    '''
    # Get initial cutoffs:
    cutoffs0 = np.linspace(0,1,(N+1))
        # Need to plug into the inverse ecdf

    cutoffs = mquantiles(a=data, prob=cutoffs0, alphap=1.0/3.0, betap=1.0/3.0)
    # mquantiles(a, prob=[0.25, 0.5, 0.75], alphap=0.4, betap=0.4, axis=None, limit=())
    #  (alphap,betap) = (1/3, 1/3): p(k) = (k-1/3)/(n+1/3): Then p(k) ~ median[F(x[k])]. The resulting quantile estimates are approximately median-unbiased regardless of the distribution of x. (R type 8)

    # Set infinite upper and lower cutoffs:
    cutoffs[0] = -np.inf
    cutoffs[-1] = np.inf

    # Init containers
    EX = []
    pX = []

    for lo, hi in zip(cutoffs[:-1], cutoffs[1:]):
        bin_indx = np.logical_and(data >= lo, data < hi)
        EX.append(np.mean(data[bin_indx]))      # Should converge to correct
        pX.append(np.mean(bin_indx))            # Should also converge to proper

    EX = np.array(EX)
    pX = np.array(pX)

    return EX, cutoffs[1:-1], pX   # want to slice off the -inf and inf bin cutoffs


def calculate_mean_sacrifice_value(vinv,vhat,xvals,xprobs):
    '''
    Given the inverse of the optimal value function, a value function for an
    approximate policy, and a discrete distribution, return the expected
    sacrifice value.

    Parameters
    ----------
    vinv: univariate real-valued function: vinv:R -> R
        Inverse of the optimal value function.
    vhat: univariate real-valued function: vinv:R -> R
        Value function for approximate policy for which we will find the
        expected sacrifice value.
    xvals: np.ndarray
        Discrete points for discrete probability mass function.
    xprobs: np.ndarray
        Probability associated with each point in xvals.

    Returns
    ----------
    Eeps: float
        Mean sacrifice value.

    Nathan M. Palmer
    June 2013
    '''
    eps = lambda x: x - vinv(vhat(x))
    return np.dot(eps(xvals), xprobs)


