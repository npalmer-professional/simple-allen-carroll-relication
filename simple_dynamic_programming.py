import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from scipy.stats.mstats import mquantiles
from scipy.interpolate import InterpolatedUnivariateSpline

# ------------------------------------------------------------------------------
# ------ Define optimization "helper" functions for welfare analysis -----------
# ------------------------------------------------------------------------------
#@jit(nopython=True)

@jit
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


@jit
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

@jit
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

@jit
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

@jit    
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

@jit
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



# ==============================================================================
# Compare DP solutions
# ==============================================================================

if __name__ == '__main__':

    # Import the non-jit-ified values: 
    from basic_consumption_savings import generalized_policy_iteration as nonjit_generalized_policy_iteration
    from basic_consumption_savings import simple_income_process
    import agent_problem_params as param
    from time import time
    
    
    # ==============================================================================
    # ============================ Set up parameters ===============================
    # ==============================================================================

    # ------------------------------------------------------------------------------
    # ------------------------- Set up space ---------------------------------------
    # ------------------------------------------------------------------------------

    # Stochastic shocks
    Ypoints = param.Ypoints
    Yprobs  = param.Yprobs

    # Set up income process
    income_seed = param.income_seed
    income_generator = simple_income_process(values=Ypoints, probs=Yprobs, seed=income_seed)

    # Set up the grid:
    grid_min = 1e-3
    grid_max = 5
    grid_size = 30
    plotgrid_size = 100

    grid = np.linspace(grid_min, grid_max, grid_size)
    plotgrid = np.linspace(grid_min, grid_max, plotgrid_size)

    tol = 1e-6

    # ------------------------------------------------------------------------------
    # --------------------- Set up agent problem values ----------------------------
    # ------------------------------------------------------------------------------
    rho = param.rho
    R   = param.R
    beta = param.beta

    # Set up utility
    u = lambda c, rho=rho: c**(1.0-rho) / (1.0-rho) # Utility function
    u.prime = lambda c, rho=rho: c**(-rho)          # Add derivative attribute/method
    u.prime_inv = lambda w, rho=rho: w**(-1.0/rho)  # Add derivative attribute/method

    # Set up the transition function:
    f  = lambda x, c, z, R=R:  R*(x-c) + z  # Given 'pre-decision' (regular) state
    fa = lambda a, z, R=R: R*a + z          # Given 'post-decision' state


    # ==============================================================================
    # ======== Set up optimization functions for sacrifice values ==================
    # ==============================================================================
    t0=time()
    nonjit_optcons, nonjit_optval = nonjit_generalized_policy_iteration(u=u, f=f,
                                                                        beta=beta, grid=grid,
                                                                        tol=tol, Ypoints=Ypoints,
                                                                        Yprobs=Yprobs, v=None,
                                                                        policy=None, verbose=True)
    t1=time()

    # Run the jit-ed version once, then again to recompile
    optcons_0, optval_0 = generalized_policy_iteration(u=u, f=f,
                                                       beta=beta, grid=grid,
                                                       tol=tol, Ypoints=Ypoints,
                                                       Yprobs=Yprobs, v=None,
                                                       policy=None, verbose=False)

    # Run the jit-ed version
    t2=time()
    jit_optcons, jit_optval = generalized_policy_iteration(u=u, f=f,
                                                           beta=beta, grid=grid,
                                                           tol=tol, Ypoints=Ypoints,
                                                           Yprobs=Yprobs, v=None,
                                                           policy=None, verbose=True)
    t3=time()

    # Print out the times: 
    print("Non-jitted version took:", (t1-t0)/60, "min")
    print("Jitted version took:", (t3-t2)/60, "min")





