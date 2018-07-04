'''
create_sacrifice_value_contour_plot.py

The only purpose of this file is to:

* find sacrifice values over a grid
* structure that data in proper form
* create a contour plot

and then also:

* given a set of function parameters, create a "walk" over the contours
    * from RL
    * from PI
    * from OPI

That's it. Get to it.
'''

# Pull in lots of information:

from __future__ import division, print_function
from builtins import zip, range, object
import pickle
import numpy as np
import pylab as plt
from time import time
from scipy.stats.mstats import mquantiles
import agent_problem_params as param
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from basic_consumption_savings import simple_income_process, generalized_policy_iteration, gpi_value_estimation, generate_simulated_choice_data, find_empirical_equiprobable_bins_midpoints, calculate_mean_sacrifice_value

# ==============================================================================
# ============================ Set up parameters ===============================
# ==============================================================================

verbose = False
CALCULATE_FROM_SCRATCH = True # true to calculate sacrifice values from scratch
CREATE_PLOT = True

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

# Consumption function parameters 
mpc_min = 0.0
mpc_max = 1.0
mpc_N = 50
buffertarget_min = 1.0
buffertarget_max = 3.0
buffertarget_N = 50

# Set up points to plot
V = [0.9, 0.75, 0.5, 0.4, 0.2, 0.1, 0.05]

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


# Unclear if need:
Nbins_for_sacrifice_approx = 501
T_ergodic_periods = 1500
N_ergodic_agents = 5000
discarded_periods = 20


# ==============================================================================
# ======== Set up optimization functions for sacrifice values ==================
# ==============================================================================
t0=time()
optcons, optval = generalized_policy_iteration(u=u, f=f,
                                               beta=beta, grid=grid,
                                               tol=tol, Ypoints=Ypoints,
                                               Yprobs=Yprobs, v=None,
                                               policy=None, verbose=verbose)

inv_optval = InterpolatedUnivariateSpline(x=optval.get_coeffs(), y=optval.get_knots(), k=1)

# Simulate consumption rule and find 5%, 95% conf intervals
sim_m, sim_c, y_shocks, mvals = generate_simulated_choice_data(cons=optcons,
                                              income_generator=income_generator,
                                              periods=T_ergodic_periods, Nagents=N_ergodic_agents, R=R,
                                              discard_periods=discarded_periods, m0=None)

# Create a flattened version of m-values, for many future uses:
flat_sim_m = sim_m.ravel()
print("Find the discrete distribution of ergodic optimal m-choices")

# Find a discrete approximation to the the distribution of m-values. This
# will be used to calculate the sacrifice value for a particular approximate
# policy.
Nbins_for_sacrifice_approx = min(Nbins_for_sacrifice_approx, len(flat_sim_m)/2.0)
sim_m_mids, sim_m_bins, sim_m_probs = find_empirical_equiprobable_bins_midpoints(N=Nbins_for_sacrifice_approx, data=flat_sim_m)

sim_m_mids_small, sim_m_bins_small, sim_m_probs_small = find_empirical_equiprobable_bins_midpoints(N=7, data=flat_sim_m)
t1=time()
print("Solved optimization problem and simulated agents in",  (t1-t0)/60.0, "min")


# ==============================================================================
# ================= Set up function to create eps_bar values ===================
# ==============================================================================

# Set up the full function:
def find_epsilon_bar_sacrifice_value(EY, mpc, mbar, f, beta, u, Ypoints, Yprobs,
                                     grid, tol, inv_optval, sim_m_mids, sim_m_probs,
                                     q=0.0):
    '''
    q is the borrowing constraint.
    '''
    # Create consumption function:
    temp_cons = lambda m, EY=EY, mpc=mpc, mbar=mbar, q=q: np.maximum(np.minimum(EY + mpc*(m-mbar), m), q)

    # Find approximate value function for temp_cons:
    approx_val = gpi_value_estimation(v=u, policy=temp_cons, u=u, f=f, 
                                      beta=beta, grid=grid, tol=tol, 
                                      Ypoints=Ypoints, Yprobs=Yprobs)
                                     
    # Find the sacrifice value:
    temp_sacrifice_val = calculate_mean_sacrifice_value(vinv=inv_optval,
                                                         vhat=approx_val,
                                                         xvals=sim_m_mids,
                                                         xprobs=sim_m_probs)
    # Return sacrifice value:
    return temp_sacrifice_val


min_sim_m_mids = min(sim_m_mids)


# Set up the function which *only* takes the parameters on consump function::
def epsilon_bar(mpc_mbar, EY=1.0, f=f, beta=beta, u=u, Ypoints=Ypoints,
                Yprobs=Yprobs, grid=grid, tol=tol, inv_optval=inv_optval,
                sim_m_mids=sim_m_mids, sim_m_probs=sim_m_probs, q=0.0):

    temp_c_fxn = lambda m, EY=EY, mpc=mpc_mbar[0], mbar=mpc_mbar[1]: np.maximum(np.minimum(EY + mpc*(m-mbar), m), 0.0)

    eps_bar = find_epsilon_bar_sacrifice_value(EY=EY, mpc=mpc_mbar[0], mbar=mpc_mbar[1],
                                             f=f, beta=beta, u=u,
                                             Ypoints=Ypoints,Yprobs=Yprobs,
                                             grid=grid, tol=tol,
                                             inv_optval=inv_optval,
                                             sim_m_mids=sim_m_mids,
                                             sim_m_probs=sim_m_probs, q=0.0)
    if eps_bar < 0:
        print("eps_bar < 0  for mpc_mbar = ", mpc_mbar)
        print("Note: temporary_cons(min_sim_m_mids) = ",temp_c_fxn(min(sim_m_mids)))
        eps_bar = 1e5  # If eps_bar is < 0 then we have trickiness.

    return eps_bar



def find_epsilon_bar_sacrifice_value_MC(EY, mpc, mbar, f, beta, u, Ypoints, Yprobs,
                                     grid, tol, inv_optval, flat_sim_m,
                                     q=0.0):
    '''
    q is the borrowing constraint.
    '''

    # Create consumption function:
    temp_cons = lambda m, EY=EY, mpc=mpc, mbar=mbar: np.maximum(np.minimum(EY + mpc*(m-mbar), m), q)

    # Find approximate value function for temp_cons:
    approx_val = gpi_value_estimation(v=u, policy=temp_cons, u=u, f=f, 
                                      beta=beta, grid=grid, tol=tol, 
                                      Ypoints=Ypoints, Yprobs=Yprobs)
                                      
    # Find the sacrifice value:
    eps = lambda x, vopt_invs=inv_optval, w=approx_val: x - vopt_invs(w(x))
    temp_sacrifice_val = np.mean(eps(flat_sim_m))

    # Return sacrifice value:
    return temp_sacrifice_val


min_flat_sim_m = min(flat_sim_m)
def epsilon_bar_MC(mpc_mbar, EY=1.0, f=f, beta=beta, u=u, Ypoints=Ypoints,
                Yprobs=Yprobs, grid=grid, tol=tol, inv_optval=inv_optval,
                flat_sim_m=flat_sim_m, min_flat_sim_m=min_flat_sim_m, q=0.0):


    temp_c_fxn = lambda m, EY=EY, mpc=mpc_mbar[0], mbar=mpc_mbar[1]: np.maximum(np.minimum(EY + mpc*(m-mbar), m), 0.0)

    eps_bar = find_epsilon_bar_sacrifice_value_MC(EY=EY, mpc=mpc_mbar[0], mbar=mpc_mbar[1],
                                             f=f, beta=beta, u=u,
                                             Ypoints=Ypoints,Yprobs=Yprobs,
                                             grid=grid, tol=tol,
                                             inv_optval=inv_optval,
                                             flat_sim_m=flat_sim_m, q=0.0)
    if eps_bar < 0:
        print("eps_bar < 0  for mpc_mbar = ", mpc_mbar)
        print("Note: temporary_cons(min_sim_m_mids) = ",temp_c_fxn(min_flat_sim_m))
        eps_bar = 1e5  # If eps_bar is < 0 then we have trickiness, likely due to -np.inf util

    return eps_bar


# ==============================================================================
# ================= Set up combos of sacrifice values ==========================
# ==============================================================================
mpc_range = np.linspace(mpc_min, mpc_max, mpc_N)
buffertarget_range = np.linspace(buffertarget_min, buffertarget_max, buffertarget_N)

if CALCULATE_FROM_SCRATCH:
    # Since it is super important to not accidentally re-run everything and save
    # over the very nice contour file, manually commenting the following lines
    # will be required to run this code.

    #print "remember to un-comment this area if you *really* need to re-run everything to make the contour plot."
    
    print("================= Calculating eps-bar values from scratch ======================")
    print("***************** Note: This may take a while! *********************************")

    # Set up containers
    X, Y = np.meshgrid(mpc_range, buffertarget_range)
    Z = np.zeros( X.shape )
    zero_ctr = 0


    min_m_to_use = min_sim_m_mids # min_flat_sim_m

    t000 = time()
    theta_param_vals = []
    z_indexes = []
    mpc_mbar_list = []
    eps_bar_list = []
    total_n_iteration = len(mpc_range)*len(buffertarget_range)
    print_N_times = 20
    divosr = total_n_iteration // print_N_times
    print("Will print 20 iterations.")
    ctr = 0
    for i, mpc in enumerate(mpc_range):
        for j, mbar in enumerate(buffertarget_range):
            # Check to see if this implies zero consumption for smallest sim_m_mids:
            # Create temp consump function:
            '''
            temp_c_fxn = lambda m, EY=1.0, mpc=mpc, mbar=mbar: np.maximum(np.minimum(EY + mpc*(m-mbar), m), 0.0)
            if temp_c_fxn(min_m_to_use) <= 0:
                print "hit zero for mpc, mbar = ",mpc,  mhat
                zero_ctr += 1
                #next
            theta_param_vals.append((mpc,mbar))
            z_indexes.append((i,j))
            '''
            if ctr % divosr == 0:
                print("One step",ctr,"of",total_n_iteration)
            ctr += 1

            # Find sacrifice value, be super explicit about where this is:
            eps_bar = epsilon_bar([mpc, mbar])
            X[i, j] = mpc
            Y[i, j] = mbar
            Z[i,j] = eps_bar
            mpc_mbar_list.append((mpc, mbar))  # Create some flat lists just to be extremely clear
            eps_bar_list.append(eps_bar)       # Sam here.

    t111 = time()

    # Create interpolation:
    interpolated_surface_function = RegularGridInterpolator((mpc_range, buffertarget_range), Z)

    print("Time to find values: ", (t111-t000)/60., "min,", t111-t000, "sec")

    print("Total # rules is:",len(theta_param_vals))
    print("Total zero-values is:", zero_ctr)

    # Save values, because, come on...
    with open('epsbar_mpc_mhats_XYZV.pickle', 'wb') as a_file:
        description = r'''Description of contents:
        epsbars: list of sacrifice values, should be in same order as mpc_mhats
        mpc_mhats: list of all tuples of (mpc, mbar) used
        mpc_range: array of all mpc values in ascending order
        buffertarget_range: array of all buffertarget values in ascending order
        X, Y, Z: meshgrid arrays of mpc, mbar and ebs_bar values respectively
        v: list of floats; surface contours to plot
        
        NOTES: interpolated_surface_function creation: interpolated_surface_function = RegularGridInterpolator((mpc_range, buffertarget_range), Z)
        '''
        var = {'description':description,
               'epsbars':eps_bar_list, 'mpc_mhats':mpc_mbar_list, 
               'mpc_range':mpc_range, 'buffertarget_range':buffertarget_range,
               'interpolated_surface_function':interpolated_surface_function,
               'X':X, 'Y':Y, "Z":Z, "V":V}
        pickle.dump(var, a_file)
    
    #pass


# ==============================================================================
# ================= Find the sacrifice value of consume everything =============
# ==============================================================================

cons_spendthrift = lambda m, EY=1.0, mpc=1.0, mbar=0.0: np.maximum(np.minimum(EY + mpc*(m-mbar), m), 0.0)
#eps_bar_spendthrift = epsilon_bar_MC(mpc_mbar=[1.0, 0.0])
eps_bar_spendthrift = epsilon_bar(mpc_mbar=[1.0, 1.0])

mpc11 = 0.15789473684210525
mbar11 = 1.4210526315789473

cons_11 = lambda m, EY=1.0, mpc=1.0, mbar=0.0: np.maximum(np.minimum(EY + mpc*(m-mbar), m), 0.0)
#eps_bar_spendthrift = epsilon_bar_MC(mpc_mbar=[1.0, 0.0])
eps_bar_11 = epsilon_bar(mpc_mbar=[mpc11, mbar11])

# Check the welfare-minimizing value:
from scipy.optimize import fmin
OUTPUT = fmin(epsilon_bar, [0.2, 1.0], full_output=True)

if OUTPUT[4] != 0:
    raise Exception("OUTPUT[4] != 0 -- OUTPUT[4] == "+ str(OUTPUT[4]))
'''Returns
-------
xopt : ndarray
    Parameter that minimizes function.
fopt : float
    Value of function at minimum: ``fopt = func(xopt)``.
iter : int
    Number of iterations performed.
funcalls : int
    Number of function calls made.
warnflag : int
    1 : Maximum number of function evaluations made.
    2 : Maximum number of iterations reached.
'''

# ==============================================================================
# ================= Plotting all Sacrifice Values     ==========================
# ================= WARNING: This may take some time! ==========================
# ==============================================================================


t2 = time()

argminsacrifice = np.nanargmin(eps_bar_list)

minsacrifice = round(eps_bar_list[argminsacrifice], 4)
minsacrifice2 = np.nanmin(eps_bar_list)

min_mpc = mpc_mbar_list[argminsacrifice][0]    #min_ij[0]][0]
min_mbar = mpc_mbar_list[argminsacrifice][1]      #min_ij[1][1]




# Set up points to plot
V += [eps_bar_spendthrift*0.9999]
V += [minsacrifice*1.02, minsacrifice*1.0125, minsacrifice*1.015, minsacrifice*1.01] # Force there to be a "solid" dot...
V.sort()

plt.figure()
#CS = plt.contour(X, Y, Z, V)
CS = plt.contour(X, Y, Z, V, colors='k')
plt.clabel(CS, inline=1, fontsize=10)
#plt.title('Isosacrifice Contours')
plt.xlabel(r"Marginal propensity to consume $\kappa$")
plt.ylabel(r"Buffer stock target $\bar{m}$")
plt.text(x=min_mpc*1.06, y=min_mbar*0.95, s=minsacrifice, fontsize=10)
#plt.text(x=0.5, y=1.5, s="TESTING", fontsize=10)
#plt.show()
plt.savefig("minimal_isosacrifice_contours_epsilon_bar.pdf")
plt.close()


