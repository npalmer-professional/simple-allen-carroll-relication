'''
Explore possible parameterizations of perpetual youth.
'''

import numpy as np
from itertools import combinations
from decimal import Decimal, getcontext
from scipy.optimize import shgo

'''
running 
    

    # Set bounds: 
    bnd_eps = 1e-3 # Choosen as 1/10000
    beta_bnd = Params.DiscFacAdj_bound
    # First set the boundaries on beta-values: 
    bnd_list = []
    beta_bound_total_diff = beta_bnd[1] - beta_bnd[0]
    beta_bound_delta = beta_bound_total_diff / n_types
    lower_beta_bnd = beta_bnd[0]
    upper_beta_bnd = lower_beta_bnd + beta_bound_delta
    for n in range(1, n_types+1):
        bnd_list.append([lower_beta_bnd, upper_beta_bnd-bnd_eps])
        lower_beta_bnd = upper_beta_bnd
        upper_beta_bnd = lower_beta_bnd + beta_bound_delta
    # Now set the bounds on the fraction -- 1 less than then n_types:
    for m in range(n_types-1):
        bnd_list.append([0+bnd_eps,1-bnd_eps])
    
    # Now add the bounds on CRRA and turn into tuple:
    bnd_list.append(Params.CRRA_bound)
    bnds = tuple(bnd_list)
    
    # If n > 1, need to also set constraints: 
    if n_types > 1:
        const_list = []
        for n in range(n_types-1):
            # Constrain beta values to be greater than one another
            const_list.append({'type':'ineq', 'fun':lambda x, i=n: x[i+1] - x[i]})  # >= 0
        
        # Also caputre that sume of probs <= 1:
        def gsum(x, n=n_types):
            return 1 - np.sum(x[n:-1]) # >= 0
        const_list.append({'type':'ineq', 'fun':gsum})

        const = tuple(const_list)

bnd_list.append(Params.CRRA_bound)
bnds = tuple(bnd_list)
const_list.append({'type':'ineq', 'fun':lambda x, i=n: x[i+1] - x[i]})  # >= 0
const = tuple(const_list)

res = shgo(fxn, bounds=bnds, constraints=const, 
           iters=iters, sampling_method='simplicial', 
           options={'minimize_every_iter':True, 'disp': True})
'''



class Agent(object):

    def __init__(self, survive_probs=None, transition_probs=None):

        # Set up some placeholders here::
        self.state            = None
        self.survive_probs    = None  # probability that agent survives. 1-prob(die)
        self.transition_probs = None  # probability that agent transitions to next state 1-prob(stay)
        self.M = None
        self.alive = True

        # Update placeholders:
        if survive_probs is not  None and transition_probs is not None:
            self.set_agent_probs(survive_probs=survive_probs,
                                 transition_probs=transition_probs)

    def initialize_agent_probs(self, survive_probs, transition_probs):
        self.state    = 0
        self.survive_probs    = np.array(survive_probs)
        self.transition_probs = np.array(transition_probs)
        self.M = len(self.survive_probs)
        self.alive = True

        transition_N = len(self.transition_probs)

        if transition_N == self.M:
            assert self.transition_probs[-1] == 0
        elif transition_N == self.M-1:
            self.transition_probs = np.append(self.transition_probs,0.0)
        else:
            assert transition_N == self.M or transition_N == self.M-1, "len(self.transition_probs) != self.M or self.M-1: "+str(transition_N)+", "+str(self.M)




class SimpleSimulatePopulation(object):

    def __init__(self, N=1000, T=120, seed=None):
        self.M = None
        self.N, self.T = N, T
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        
        # generate list of agents:
        self.agents = [Agent(survive_probs=None, transition_probs=None) for i in range(self.N)]

        # Generate shocks for the population: 
        self.live_shocks   = [np.random.uniform(size=self.T) for n in range(self.N)]
        self.change_shocks = [np.random.uniform(size=self.T) for n in range(self.N)]

        # Generate container for counting agents alive:
        self.number_alive_beginning_each_age = np.zeros(self.T, dtype=np.int)
        self.number_alive_at_end_end_age     = np.zeros(self.T, dtype=np.int)
        
        self.max_error = 2**31-1  # Really big number
        

    def initialize_agent_probs(self, p_live, p_change):
        self.M = len(p_live)
        for agent in self.agents:
            agent.initialize_agent_probs(survive_probs=p_live, transition_probs=p_change)

    @property
    def N_agents_alive(self):
        return sum((agent.alive for agent in self.agents))

    @property
    def agent_states(self):
        return [agent.state for agent in self.agents]

    def simulate_actuarial_table(self, p_live, p_change):
        # Set all agents: 
        self.initialize_agent_probs(p_live, p_change)
        
        # Loop and flip the coin for everyone:
        for t in range(self.T):
            #agents_alive_this_period = 0
            agents_alive_beginning_of_period = 0
            agents_alive_end_of_period = 0
            for n in range(self.N):
                agent = self.agents[n]
                agent_live_shock   = self.live_shocks[n]
                agent_change_shock = self.change_shocks[n]

                # Update beginning of period counter before "enact" period
                agents_alive_beginning_of_period += agent.alive
                if agent.alive:
                    i = agent.state

                    # Flip coin to see if agent lives:
                    agent.alive = agent_live_shock[t] <= agent.survive_probs[i]

                    # If agent alive, flip coin to see if change state:
                    if agent.alive:
                        if agent_change_shock[t] <= agent.transition_probs[i]:
                            # Then agent is shifting to the next state, recalling
                            # that can never go over the max state
                            agent.state = min(agent.state + 1, self.M-1)
                # Update end of period counter after "enact" period
                agents_alive_end_of_period += agent.alive

            self.number_alive_beginning_each_age[t] = agents_alive_beginning_of_period
            self.number_alive_at_end_end_age[t]     = agents_alive_end_of_period
        return self.number_alive_beginning_each_age

    def set_expected_actuarial_table(self, expected_actuarial_table):
        self.expected_actuarial_table = expected_actuarial_table

    def __call__(self, p_live_p_change):
        # Assume M states, M-1 changes:
        M = int( (len(p_live_p_change) + 1) / 2)
        
        p_live   = p_live_p_change[:M]
        p_change = p_live_p_change[M:]

        # Check if any issues with prob live or prob change:
        if np.any(p_live < 0) or np.any(p_live > 1):
            return self.max_error            
        if np.any(p_change < 0) or np.any(p_change > 1):
            return self.max_error            
        # actually let's let this play out itself. Hmmm. 
        #if np.any(np.diff(p_live) > 0):
        #    return self.max_error
        
        # Ok, run the sim and return error:        
        vals = self.simulate_actuarial_table(p_live=p_live, p_change=p_change)
        return sum( (vals - self.expected_actuarial_table)**2 )


class SimulatePopulation(object):

    def __init__(self, expected_number_alive_per_period, 
                 N=1000, T=120, one_transition_prob=False, seed=None):
        ''' Initilize population of finite perpetual youth agents.

        Parameters
        M : int
            Number of epochs in life
        prob_survive_per_cohort : array-like
            Length M+1, probability of surviving to the next period. Represents
            M states of life, with the final entry being 0 (has died).
        prob_transition_to_next_epoch : array-like
            Length M, probability that transition to the next aging group. 
            Represents M-1 probablities, plus the final probability is 0
            (agent has died and no longer transitioning).

        '''
        self.M = None #M # Number of "epochs" in the agents' lives
        self.N = N # Number of agents to simulate
        self.T = T # Max age, start counting at 0
        
        # Save this. CAn reset later
        self.one_transition_prob = one_transition_prob
        
        # Set indeces to count things:
        self.number_alive_beginning_each_age = None
        self.number_alive_at_end_end_age = None
        self.array_alive_each_period = None
        self.agent_indeces = None
        self.agent_state_i = None
        # Init all the above:
        self.init_agent_arrays()
        
        # Set the number expected alive:
        self.expected_number_alive_per_period = expected_number_alive_per_period / expected_number_alive_per_period[0] * N 
        assert len(expected_number_alive_per_period) == self.T, "len(expected_number_alive_per_period) != self.T:"+str(len(expected_number_alive_per_period))+", "+str(self.T)
        
        # Use the expected number alive to create a "maximum wrongness error"
        # to return when the prbs passed in are incorrect:
        self.max_error = sum(self.expected_number_alive_per_period**2)
        
        # Set RNG:
        self.seed = seed
        self.rng=np.random.RandomState(self.seed)

        # Create a list of shocks of agent-length, each period:
        self.live_shocks   = [self.rng.uniform(size=self.N) for t in range(self.T)]
        self.change_shocks = [self.rng.uniform(size=self.N) for t in range(self.T)]


    def init_agent_arrays(self):
        # Set indeces to count things:
        self.number_alive_beginning_each_age = np.zeros(self.T)
        self.number_alive_at_end_end_age = np.zeros(self.T)
        self.array_alive_each_period = np.ones(self.N, dtype=np.bool)
        self.agent_indeces = np.arange(self.N, dtype=np.int)
        self.agent_state_i = np.zeros(self.N, dtype=np.int)


    def simulate_actuarial_table(self, p_live, p_change):
        # Init agent arrays:
        self.init_agent_arrays()
        # Set the M:
        self.M = len(p_live)
        if len(p_change) == len(p_live)-1:
            # Then add a last element of p_change with 0:
            p_change = np.append(p_change, [0])
        for t in range(self.T):
            # Find all agents alive this period: 
            i_alive = self.agent_indeces[self.array_alive_each_period]
            self.number_alive_beginning_each_age[t] = sum(self.array_alive_each_period)
            # Find the prob-cutoff for all living agents:
            #print(t)
            #print(i_alive)
            #print(self.agent_state_i[i_alive])
            #print(self.agent_state_i)
            #print('---------')
            live_cutoff   = p_live[ self.agent_state_i[i_alive] ]
            
            # Flip the "do they live" coin:
            live_shocks = self.live_shocks[t]
            agents_live_bool = live_shocks[i_alive] <= live_cutoff
            
            # Update the "alive" array indicator -- save only the index that 
            # lived:
            i_alive = i_alive[agents_live_bool]
            
            # Set alive_each_period to all false, and update only the alive index:
            self.array_alive_each_period = np.zeros(self.N, dtype=np.bool)
            self.array_alive_each_period[i_alive] = True
            
            # Now use the updated "alive" index to flip coins for whether to
            # transition:
            #change_cutoff   = p_change[i_alive]
            change_shocks = self.change_shocks[t]
            agents_change_bool = change_shocks[i_alive] <= p_change[self.agent_state_i[i_alive]]
            agents_not_at_max_epoch_bool = self.agent_state_i[i_alive] < self.M-1
            i_change_epoch = np.logical_and(agents_change_bool, agents_not_at_max_epoch_bool)
            
            i_alive_and_changed = i_alive[i_change_epoch]
            
            # Update the epoch for appropriate agents:
            self.agent_state_i[i_alive_and_changed] += 1
            
            # Now count the number of agents alive at the end of the period
            self.number_alive_at_end_end_age[t] = sum(self.array_alive_each_period)

        return self.number_alive_beginning_each_age

    def __call__(self, plive_pchange):
    
        if self.one_transition_prob:
            self.M = len(plive_pchange) - 1
            # Set the values:
            p_live = np.array(plive_pchange[:self.M])  # Grab the first M values which should the be prob live each period
            # Grab the transition values:
            p_change = np.repeat(plive_pchange[-1], self.M-1)
        else:
            assert len(plive_pchange) % 2 == 1, "Since self.one_transition_prob == False, len(plive_pchange) should be odd."
            self.M = int( (len(plive_pchange) + 1)/2)
            # Set the values:
            p_live = np.array(plive_pchange[:self.M])  # Grab the first M values which should the be prob live each period
            # Grab the transition values:
            p_change = np.array(plive_pchange[self.M:])
            

        # Confirm that transition values are either length M-1 or length 1
        if len(p_change) == 1:
            if self.M > 2:
                # Then need to repeat p_change proper number of times:
                p_change = np.array([p_change[0] for m in range(self.M-1)])
        assert len(p_change) == 1 or len(p_change) == self.M-1, "p_change is wrong length"

        # If probabilities are less than zero, or greater than one, or the 
        # survival probabilities are not decreasing, return self.max_error
        survival_probs_weakly_decreasing = np.all(np.diff(p_live) <= 0)
        some_probs_lt_0 = np.any(p_live < 0) or np.any(p_change < 0)
        some_probs_gt_1 = np.any(p_live > 1) or np.any(p_change > 1)
        if some_probs_lt_0 or some_probs_gt_1 or not survival_probs_weakly_decreasing:
            return self.max_error
        else:
            n_sim = self.simulate_actuarial_table(plive_pchange[:self.M], plive_pchange[self.M:])
            return sum( (n_sim - self.expected_number_alive_per_period)**2)




# ALTERNATE: need to write out all transition probabilities via counting
# exercise.
class FinitePerpetualYouthCombinatoric(object):
    '''
    Timing of the perpetual youth setup:
    
    Assume: 
        p_live[i]: prob don't die in epoch i
        p_change[i]: prob transition from epoch i to i+1
    
        Note: Should probably start with just equal prob of transition from one 
        epoch to the next (where final prob_transition == 0). 
    
    Timing:
    
    - "wake up" in period t, epoch i
    - "flip coin", p_live[i], to see if survive in period t
    - if survive, "flip coin" p_change[i], to see if transition to epoch i+1
        - if transition, then i += 1
    
    Add one more thing: we'd like to fit this to *actual life/death actuarial
    tables* from real life, for ages...say 0 to 119 (120 years, what we have in
    the actuarial tables I've pulled down). 
    
    So here is what to do: 
    
    - At each age, calculate the prob death at this age. This is going to be a 
      complicated thing -- going to need to calculate all the ways that could 
      have survived to this age.
      NOTE that have done this on paper -- need to **go find that paper** and 
      implement in code!
    - for each age t and total number of epochs M, 
        possible_max_epochs = min(t, M)
    - *Now* need to build the combinatorial probability that survived up to this 
      age. Going to need to build out list of combinatorial probs and sum across
      them. ("this way" or "this way" or "this way"...)
      
    Thus at each (t,i) point -- at time t, in epoch i -- we can calculate
    the probability that we got here.
    

    Example:    
    Assume the following p_live, p_change:
    
        epoch i =       0,      1,      2,      3,      4
        p_live  =    0.99,   0.90,   0.75,   0.65,      0   
        p_chge  =    0.25,   0.15,   0.20,      0

    Assume the following "case" or "combo:"  (2,3,5)  # Times at which transition happened
       ...and current age of t = 10
    So first transition happens in period 2, then next in 3, then next transition
    in period 5, then live in this state until current age of t=10

    So here's the table:
    
    t ∈ {0,1,2}; i == 0
    .99*(1-.25) * .99*(1-.25)       # no transition in first two periods
                * .99*(.25)         # survive and transit in period t=2
    t = {3}; i == 1
   *.90*(.15)                       # survive and transit in period t=3
    t = {4,5}; i == 2
   *0.75*(1-.2) * 0.75*.2           # survive and transit in period t=5
    t = {6,7,8,9,10}; i == 3        
   *(0.65)**(5)                     # No more transit prob; just survive to t=10
    
    Let's turn this into code.
    p = 1
    t0 = 0
    t_age_eop = whatever it is
    for i,t in enumerate(case):
        p *= p_live[i]**(t-t0+1) * (1-p_chge[i])**(t-t0) * p_chge[i]
        t0 = t+1
    # Recall that should be at the next point now, so want i+1
    p *= p_live[i+1]**(t_age_eop - t)

    # i=0, t=2, t0=0  .99^3 * (1-.25)^2 * .25
    # i=1, t=3, t0=3  .9^1  * (1-.15)^0 * .15
    # i=2, t=5, t0=4  .75^2 * (1-.20)^1 * .20

    '''


    def __init__(self, M, T, expected_prob_alive_per_period, 
                 decimal_precision=100, only_one_transition_prob=True):
        ''' Initilize population of finite perpetual youth agents.

        Parameters
        M : int
            Number of epochs in life
        prob_survive_per_cohort : array-like
            Length M+1, probability of surviving to the next period. Represents
            M states of life, with the final entry being 0 (has died).
        prob_transition_to_next_epoch : array-like
            Length M, probability that transition to the next aging group. 
            Represents M-1 probablities, plus the final probability is 0
            (agent has died and no longer transitioning).

        '''
        self.M = M # Number of "epochs" in the agents' lives
        
        self.M = M # Number of epochs
        self.T = T # Max number of periods; prob survive =0 after this
        self.only_one_transition_prob = only_one_transition_prob  # Flag: should we only use one constant prob of transition?
        #self.survive_probs = [None for t in range(T)] # Will fill in as go
        #self.survive_probs_lists = []
        #self.survive_probs = np.zeros(self.T, dtype=np.float128)
        #self.survive_probs_per_epoch = np.zeros(self.M)           # Going to try something with high precision...
        #self.transition_probs_between_epoch = np.zeros(self.M-1)
        #self.survive_probs_per_epoch = [None for m in range(M)]
        #if self.only_one_transition_prob:
        #    self.transition_probs_between_epoch = None
        #else:
        #    self.transition_probs_between_epoch = [None for m in range(M-1)]

        self.max_error= max(2**31-1, sum((100*x**2 for x in expected_prob_alive_per_period)))
        getcontext().prec = decimal_precision # Set precision for Decimal
        self.expected_prob_alive_per_period = [Decimal(expected_prob_alive_per_period[i]) for i in range(len(expected_prob_alive_per_period))]
        
        assert len(self.expected_prob_alive_per_period) == self.T, "len(self.expected_prob_alive_per_period) != self.T: "+str(len(self.expected_prob_alive_per_period))+", "+str(self.T)
        
        
        # Set up place to hold most recently calculated probs:
        self.p_alive_each_age = None
        
        # Need to 
        #self.survive_probs_per_epoch = np.zeros(self.M+1)
        #self.transition_probs_between_epoch = np.zeros(self.M)

    def calculate_survival_probs(self, p_live, p_chge):
        
        # Set the first survival prob:
        survive_probs = []
        survive_probs.append(Decimal(p_live[0]))
        survive_probs_lists = []

        for t in range(1, self.T):
            possible_m = min(t, self.M-1)+1
            '''  
            For each possible value of m, need to enumerate the possible ways
            we got here. For example, for t == 1, we have 2 possible states:
                initial state
                one transistion
            For t == 2, we have 3 possible states:
                initial state
                one transition
                two transitions
            For each *possible_m*, want to count all the ways that this 
            could get filled in: 
                transition 1 period: ___
                transition 2 period: ___
                ...
                transition m period: ___

            Re-write as :   ____, ____,  ..., ______  
            and note that filling them in can be done in a for loop, and in 
            fact, each entry tells us what all the enumerations of the next 
            entry must be.

            REMEMBER how combinations work:
            from intertools import combinations

            '''
            total_prob_age = 0
            all_probs_at_current_age = []
            for m in range(possible_m):
                # Fetch all possible ages at which we get m transition 
                all_ordered_combos = combinations(range(t+1), m)
                # For each of these agen-combos, need to calculate
                # the probability that got here:
                for case in all_ordered_combos:
                    probs = []
                    #p = 1
                    t0 = 0
                    tt = t
                    i = 0  # For some of these there are no cases but need i defined 
                    t_age_eop = t
                    for i,tt in enumerate(case):
                        #p *= p_live[i]**(tt-t0+1) * (1-p_chge[i])**(tt-t0) * p_chge[i]
                        probs.append(Decimal(p_live[i])**(tt-t0+1) * Decimal((1-p_chge[i]))**(tt-t0) * Decimal(p_chge[i]))
                        t0 = tt+1
                    # Recall that should be at the next point now, so want i+1
                    #p *= p_live[i+1]**(t_age_eop - tt)
                    #if i+1 == self.M:
                    #    # Then we've "moved past" the 
                    #    probs.append(Decimal(0))
                    #else:
                    probs.append(Decimal(p_live[i+1])**(t_age_eop - tt))
                total_prob_age += sum(probs)
                all_probs_at_current_age.append(sum(probs))
            survive_probs.append(total_prob_age)
            survive_probs_lists.append(all_probs_at_current_age.copy())
        self.p_alive_each_age = survive_probs.copy()
        return survive_probs, survive_probs_lists
    
    def __call__(self, p_live_p_change):
        '''Take in prob_survival_per_cohort, prob_transition_to_next_epoch
        and append a zero to the end of each. Get back the number_alive, 
        and return the sum of squared errors between that and an actual 
        life-table.'''
        # Set the values:
        p_live = np.array(p_live_p_change[:self.M])  # Grab the first M values which should the be prob live each period
        # Grab the transition values:
        p_change = np.array(p_live_p_change[self.M:])

        # Confirm that transition values are either length M-1 or length 1
        if len(p_change) == 1:
            if self.M > 2:
                # Then need to repeat p_change proper number of times:
                p_change = np.array([p_change[0] for m in range(self.M-1)])
        assert len(p_change) == 1 or len(p_change) == self.M-1, "p_change is wrong length"

        # If probabilities are less than zero, or greater than one, or the 
        # survival probabilities are not decreasing, return self.max_error
        survival_probs_weakly_decreasing = np.all(np.diff(p_live) <= 0)
        some_probs_lt_0 = np.any(p_live < 0) or np.any(p_change < 0)
        some_probs_gt_1 = np.any(p_live > 1) or np.any(p_change > 1)
        if some_probs_lt_0 or some_probs_gt_1 or not survival_probs_weakly_decreasing:
            return self.max_error
        else:
            # Calculate the survival probabilities
            # Run the calculation
            p_survive, p_survive_lists = self.calculate_survival_probs(p_live=p_live,
                                                                       p_chge=p_change)
            return np.float64(sum( ( (p_survive[i] - self.expected_prob_alive_per_period[i])**2 for i in range(self.T) )))
            



#                for datecombo in all_ordered_combos:
#                    temp_prob = 1
#                    for i, date in enumerate(datecombo):
#                        temp_prob *= 
#                        survival_prob_per_epoch[i]**date * (1 - transition_probs_between_epoch[i])**date
#                        # Where at: here. REMEBER, this is what is happening:
#                        # Survived *and* didn't transition until date "date,"
#                        # then survived and hit the "transition" prob. 
#                        # ...then repeat until hit all the dates / get to current t. 
#                        # So need to chew on some more and get down. 
'''
Remember the world being modelled:

- there are 3 "epochs", say
- each period flip two coins 
    - one coin says whether live or die -- if live great, continue
    - next coin says whether transition to the next epoch
- I *think* (though not clear!) that this should result in a world in 
  which agents solve for three infinite-horizon-style consumption functions. 
  The same reasoning applies here as with single-perpetual-youth-epoch model:
  from one period to the next, the probabilities are all iid.
  Since the probabilities are all iid, there *should* be something like a 
  perpetual-you-infinite-horizon-style setup that falls out of this -- but 
  for the number of epoches the agent faces. 
  NEED TO THINK ABOUT AND WRITE UP. And very likely need to write up with the
  same style of approach as use for the standard finite horizon model: 
    "in final period, *know* the continuation value is 0, or live forever. So 
     solve this model."
    "in next-to-final epoch, *know* the continuation value for getting the 
     'get older' draw. And know the value of keep on living one more period..."
  ...doe this work to just "walk backards" like this? I *think* so, but need to 
  see.
  HMMMMMMMM. 
  
  NOTE that may need to 
  restrict the model such that agents only solve for three consumption 
  functions -- like a "calvo" model of HH expectation formation. 


'''
                                

 
        





if __name__ == '__main__':

    # Import 
    import pandas as pd
    from scipy.optimize import minimize, shgo
    from time import time
    import matplotlib.pyplot as plt

    # Read in the actuarial data, set up the optimization problem, and 
    # Read in this table: https://www.ssa.gov/oact/STATS/table4c6.html
    df = pd.read_csv("lifetable.csv")

    # Pull out the comparison:
    # Purposefully grab only age 20 to 119:
    male_numbers = df['Male_Number_of_lives'].values[20:90]
    female_numbers = df['Female_Number_of_lives'].values[20:90]
    date_range = np.array(range(20,90))

    ## Set up the processes:
    seed = 98765 #1234509876
    N = 1000
    male_numbers_adjusted = male_numbers/male_numbers[0] * N
    female_numbers_adjusted = female_numbers/female_numbers[0] * N

    # Set up male sime
    male_process_sim = SimpleSimulatePopulation(N=N, T=len(male_numbers), 
                                                seed=seed)
    male_process_sim.set_expected_actuarial_table(male_numbers_adjusted)
    
    # Set up female sim
    female_process_sim = SimpleSimulatePopulation(N=N, T=len(female_numbers), 
                                                  seed=seed)
    female_process_sim.set_expected_actuarial_table(female_numbers_adjusted)

    # Run once and time it
    t0=time()
    eps = 5e-4   #0.000000001
    #x_live, x_change = np.array([1-eps, 1-5*eps, 0.9, 0.20]), np.array([5*eps, 5*eps, 0.25]
    #x_live, x_change = np.array([9.90200788e-01, 8.83519027e-01, 2.00124274e-01]), np.array([5.14612086e-04,
    #   2.56629579e-01]) 
    x_live = 1-df['Male_Death_probability'].values[20:]
    
    x_live = np.array([0.9988, 0.9987, 0.9985, 0.9985, 0.9984, 0.9984, 0.9984, 0.9983, 0.9983, 0.9982, 
                       0.9982, 0.9982, 0.9981, 0.9981, 0.998 , 0.9979,0.9979, 0.9978, 0.9977, 0.9977, 
                       0.9976, 0.9975, 0.9973, 0.9972,0.997 , 0.9968, 0.9965, 0.9962, 0.9959, 0.9954, 
                       0.995 , 0.9945,0.994 , 0.9934, 0.9928, 0.9922, 0.9915, 0.9908, 0.9901, 0.9893,
                       0.9885, 0.9876, 0.9867, 0.9859, 0.9851, 0.9842, 0.9831, 0.9819,0.9805, 0.9788, 
                       0.9769, 0.9747, 0.9724, 0.9699, 0.9672, 0.964 ,0.9604, 0.9565, 0.9523, 0.9476, 
                       0.9423, 0.9361, 0.9292, 0.9216, 0.913 , 0.9034, 0.8926, 0.8805, 0.8671, 0.8524, 
                       0.8363, 0.8189,0.8002, 0.7802, 0.7591, 0.7381, 0.7178, 0.6984, 0.6806, 0.6646,
                       0.6478, 0.6302, 0.6117, 0.5923, 0.5719, 0.5505, 0.5281, 0.5045,0.4797, 0.4537, 
                       0.4264, 0.3977, 0.3676, 0.3359, 0.3027, 0.2679,0.2313, 0.1928, 0.1525, 0.1101])




    # WHERE AT: Take this and fix it, and have optimizer find the "prob change" 
    # that gets the closest to what we want.
    x_live = np.array([0.9977, 
                       0.9893,
                       0.9476, 
                       0.6646,
                       0.1101])






    """
    x_live = np.array([0.9982, 
                       0.9977, 
                       0.9954, 
                       0.9893,
                       0.9788, 
                       0.9476-.2, 
                       0.8524-.2, 
                       0.6646-.2,
                       0.4537-.2, 
                       0.1101-.1])"""


    
    '''
    x_live = np.array([0.9936, 0.9996, 0.9997, 0.9998, 0.9998, 0.9998, 0.9999, 0.9999,
                       0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9998, 0.9997, 0.9996,
                       0.9994, 0.9993, 0.9991, 0.999 , 0.9988, 0.9987, 0.9985, 0.9985,
                       0.9984, 0.9984, 0.9984, 0.9983, 0.9983, 0.9982, 0.9982, 0.9982,
                       0.9981, 0.9981, 0.998 , 0.9979, 0.9979, 0.9978, 0.9977, 0.9977,
                       0.9976, 0.9975, 0.9973, 0.9972, 0.997 , 0.9968, 0.9965, 0.9962,
                       0.9959, 0.9954, 0.995 , 0.9945, 0.994 , 0.9934, 0.9928, 0.9922,
                       0.9915, 0.9908, 0.9901, 0.9893, 0.9885, 0.9876, 0.9867, 0.9859,
                       0.9851, 0.9842, 0.9831, 0.9819, 0.9805, 0.9788, 0.9769, 0.9747,
                       0.9724, 0.9699, 0.9672, 0.964 , 0.9604, 0.9565, 0.9523, 0.9476,
                       0.9423, 0.9361, 0.9292, 0.9216, 0.913 , 0.9034, 0.8926, 0.8805,
                       0.8671, 0.8524, 0.8363, 0.8189, 0.8002, 0.7802, 0.7591, 0.7381,
                       0.7178, 0.6984, 0.6806, 0.6646, 0.6478, 0.6302, 0.6117, 0.5923,
                       0.5719, 0.5505, 0.5281, 0.5045, 0.4797, 0.4537, 0.4264, 0.3977,
                       0.3676, 0.3359, 0.3027, 0.2679, 0.2313, 0.1928, 0.1525, 0.1101])  
 


 
    x_live = np.array([0.9997, 0.9998, 0.9998, 0.9998, 0.9999, 0.9999,   # 0
                       0.9999, 0.9999, 0.9999, 0.9998, 0.9997, 0.9996,
                       0.9991, 0.999 , 0.9988, 0.9987, 0.9985, 0.9985,   # (5-fold 0, i=24)
                       0.9984, 0.9983, 0.9983, 0.9982, 0.9982, 0.9982,   
                       0.998 , 0.9979, 0.9979, 0.9978, 0.9977, 0.9977,   # 
                       0.9973, 0.9972, 0.997 , 0.9968, 0.9965, 0.9962,   # 5    (5-fold 1, i=48)
                       0.995 , 0.9945, 0.994 , 0.9934, 0.9928, 0.9922,
                       0.9901, 0.9893, 0.9885, 0.9876, 0.9867, 0.9859,
                       0.9831, 0.9819, 0.9805, 0.9788, 0.9769, 0.9747,   #  (5-fold 2, i=72)
                       0.9672, 0.964 , 0.9604, 0.9565, 0.9523, 0.9476,   # 
                       0.9292, 0.9216, 0.913 , 0.9034, 0.8926, 0.8805,   # 10
                       0.8363, 0.8189, 0.8002, 0.7802, 0.7591, 0.7381,
                       0.6806, 0.6646, 0.6478, 0.6302, 0.6117, 0.5923,
                       0.5281, 0.5045, 0.4797, 0.4537, 0.4264, 0.3977,
                       0.3027, 0.2679, 0.2313, 0.1928, 0.1525, 0.1101])  # 14






    x_live = np.array([np.mean([0.9936, 0.9996, 0.9997, 0.9998, 0.9998, 0.9998, 0.9999, 0.9999]),
                      np.mean([0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9998, 0.9997, 0.9996]),
                      np.mean([0.9994, 0.9993, 0.9991, 0.999 , 0.9988, 0.9987, 0.9985, 0.9985]),
                      np.mean([0.9984, 0.9984, 0.9984, 0.9983, 0.9983, 0.9982, 0.9982, 0.9982]),
                      np.mean([0.9981, 0.9981, 0.998 , 0.9979, 0.9979, 0.9978, 0.9977, 0.9977]),
                      np.mean([0.9976, 0.9975, 0.9973, 0.9972, 0.997 , 0.9968, 0.9965, 0.9962]),
                      np.mean([0.9959, 0.9954, 0.995 , 0.9945, 0.994 , 0.9934, 0.9928, 0.9922]),
                      np.mean([0.9915, 0.9908, 0.9901, 0.9893, 0.9885, 0.9876, 0.9867, 0.9859]),
                      np.mean([0.9851, 0.9842, 0.9831, 0.9819, 0.9805, 0.9788, 0.9769, 0.9747]),
                      np.mean([0.9724, 0.9699, 0.9672, 0.964 , 0.9604, 0.9565, 0.9523, 0.9476]),
                      np.mean([0.9423, 0.9361, 0.9292, 0.9216, 0.913 , 0.9034, 0.8926, 0.8805]),
                      np.mean([0.8671, 0.8524, 0.8363, 0.8189, 0.8002, 0.7802, 0.7591, 0.7381]),
                      np.mean([0.7178, 0.6984, 0.6806, 0.6646, 0.6478, 0.6302, 0.6117, 0.5923]),
                      np.mean([0.5719, 0.5505, 0.5281, 0.5045, 0.4797, 0.4537, 0.4264, 0.3977]),
                      np.mean([0.3676, 0.3359, 0.3027, 0.2679, 0.2313, 0.1928, 0.1525, 0.1101])])


    x_live = np.array([ 0.9999,   # 0
                        0.9996,
                        0.9985,   # (5-fold 0, i=24)
                        0.9982,   
                        0.9977,   # 
                        0.9962,   # 5    (5-fold 1, i=48)
                        0.9922,
                        0.9859,
                        0.9747,   #  (5-fold 2, i=72)
                        0.9476,   # 
                        0.8805,   # 10
                        0.7381,
                        0.5923,
                        0.3977,
                        0.1101])  # 14

    '''
    
    
    
    # WHERE AT: Take this and fix it, and have optimizer find the "prob change" 
    # that gets the closest to what we want.
    x_live = np.array([0.9977, 
                       0.9893,
                       0.9476, 
                       0.6646,
                       0.1101])
    x_live = np.repeat(0.97,5)
    x_live = np.array([0.9999, 0.825])

    x_chng = np.ones(len(x_live)-1) * 1/5
    #x_chng = np.ones(len(x_live)-1)

    x_chng[0] = 4/100
    #x_chng[1] = 15/100
    #x_chng[2] = 50/100
    #x_chng[3] = 90/100



    '''
    x_chng[0] = 0.25/10
    x_chng[1] = 0.5/10
    x_chng[2] = .75/10
    x_chng[3] = 1./10
    x_chng[4] = 1.5/10
    x_chng[5] = 2.5/10
    x_chng[6] = 3/10
    x_chng[7] = 3.5/10
    x_chng[8] = 4/10
    '''
    x_live_chunks = [None]*5
    x_live_chunks[0] = x_live[:24]
    x_live_chunks[1] = x_live[24:48]
    x_live_chunks[2] = x_live[48:72]
    x_live_chunks[3] = x_live[72:96]
    x_live_chunks[4] = x_live[96:120]
 
    print("SUM SQ ERROR:",male_process_sim(np.append(x_live, x_chng)))
 
 
    '''Recall:
    
    - poisson rate lambda
    - *wait times* is exponential distribution with mean 1/lambda
    
    So -- use this to choose wait times per period!
    
    Chop the time range into 5 equal bins (prob-spaced if initial doesn't work)
    and calculate roughly the expected value of survived agents per bin. Yes?
    
    
    In [111]: i0=0; 
     ...: for i1 in [24, 48, 72, 96, 120]: 
     ...:     print(np.mean(df['Male_Number_of_lives'].values[i0:i1])/df['Male_Number_of_lives'].values[i0]) 
     ...:     i0=i1 
     ...:                                                                                                                                              
    0.9909954166666667
    0.9782939619183383
    0.9084839346635705
    0.5572442089580931
    0.1315184340554922
    
    
    
    WHERE AT: HERE!
    '''
 
 
 
    # Good for i = 0, i=1   0.99991
    #x_live = np.array([1.0, 0.995, 0.057, 0.01])
    #x_chng = np.array([1/300, 1/2, 1/1.05]) 
    # This gets reasonable ages between 0, 55:
    #x_live = np.array([0.9999, 0.9994, 0.9750, 0.9624, 0.7178])
    #x_chng = np.array([0.0263, 0.0092, 0.0195, 0.6]) 
    # Older good fit:
    #x_live = np.array([0.9999, 0.9974, 0.9959, 0.9724, 0.7178])
    #x_chng = np.array([0.0263, 0.0022, 0.0213, 0.6]) 

    # Give big dip early:
    #x_live = np.array([0.9999, 0.95, 0.9, 0.25])
    #x_chng = np.array([0.05, 0.1, 0.75]) 



    # x_chng set by figuring out x such that:  x*max = min ->  x = min/max
    # i=0
    # print(max(x_live_chunks[i])); print(round(1-min(x_live_chunks[i])/max(x_live_chunks[i]),4)) 
    # x_live = np.array([0.9999, 0.9984, 0.9959, 0.9724, 0.7178])
    # x_chng = np.array([0.0063, 0.0022, 0.0213, 0.2410, 0.8466]) 
 
 
 
    #print("states:",male_process_sim.agent_states)

    #assert False, "Stop here."

    # Parameters:
    #x0 = np.array([.99, 0.95, .9, 0.66, .11,  0.2, .2, .2, .2])
    #x0 = np.array([0.99993952, 0.99977869, 0.9993099 , 0.99778453, 0.98001182, 0.965,
    #               0.07978252, 0.12834025, 0.16523709, 0.22208905, .3216])
    x0 = np.array([.99, .95, .66,
                   .10, .20])
    
    M_opt = int( (len(x0)+1)/2)


    actuarial_table = male_process_sim.simulate_actuarial_table(x0[:M_opt],
                                                                x0[M_opt:])
                                                                
    #    actuarial_table = male_process_sim.simulate_actuarial_table(np.array([1-eps, 0.9, 0.20]), np.array([eps, 0.25])) # gets ages 0->20
    t1=time()
    print("Took", (t1-t0)/60, "min")
    print(actuarial_table)
    for k in range(len(x0[:M_opt])):
        print("mean(states == "+str(k)+":", np.mean( np.array(male_process_sim.agent_states) == k))
    
    plt.plot(date_range, male_numbers_adjusted, color='tab:orange'); plt.plot(date_range, male_process_sim.number_alive_beginning_each_age); plt.show()

    #assert False, "Stop here."

    bnd_list = []
    eps = 1e-4
    for x in x0:
        bnd_list.append((eps,1-eps)) 
    bnds = tuple(bnd_list)
    #const_list.append({'type':'ineq', 'fun':lambda x, i=n: x[i+1] - x[i]})  # >= 0
    #const = tuple(const_list)
    
    opt_method = 'simplicial'
    iters = 3
    # ['simplicial','Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact']
    t2 = time()
    if opt_method == 'simplicial':
        res_m = shgo(male_process_sim, bounds=bnds, #constraints=const, 
                     iters=iters, sampling_method='simplicial', 
                     options={'minimize_every_iter':True, 'disp': True})
    else:    
        res_m = minimize(male_process_sim,x0=x0, method=opt_method)
    t3 = time()
    #res_f = minimize(male_process, optimizer_method=opt_method)
    #t2 = time()

    male_process_sim.simulate_actuarial_table(res_m.x[:M_opt], res_m.x[M_opt:])
    
    plt.plot(date_range, male_numbers_adjusted, color='tab:orange'); plt.plot(date_range, male_process_sim.number_alive_beginning_each_age); plt.show()


    print("Optimization took:",(t3-t2)/60,"min")
    
    #male_process_sim.simulate_actuarial_table(res_m.x[:M_epochs], res_m.x[M_epochs:])

    # plt.plot(male_process_sim.expected_number_alive_per_period);plt.plot(male_process_sim.number_alive_beginning_each_age); plt.show()


    import pickle
    results_and_time = {'t0':t0, 't1':t1, 'res':res_m}

    with open("optimize_results.pickle",'wb') as f: 
        pickle.dump(results_and_time, f) 

    #male_process_sim

'''    #SimulatePopulation(object):
    #
    #def __init__(self, M, expected_number_alive_per_period, 
    #             N=1000, T=120, seed=None):
    # Everything above is the "simualtion" version, which is basically intractable
    # with current-speed computer
    # So change it to "directly calcualte the probs:"

    # NOTE: STILL NEED TO DO TESTING!

    
    male_prob_alive_each_age = df['Male_Number_of_lives'].values / df['Male_Number_of_lives'].values[0]
    female_prob_alive_each_age = df['Female_Number_of_lives'].values / df['Female_Number_of_lives'].values[0]
    male_process = FinitePerpetualYouthCombinatoric(M=3, T=120, expected_prob_alive_per_period=male_prob_alive_each_age, 
                                                    decimal_precision=100,
                                                    only_one_transition_prob=True)

    #def __init__(self, M, T, expected_prob_alive_per_period, 
    #             decimal_precision=100, only_one_transition_prob=True):
    # Minimize it:
    opt_method = 'Nelder-Mead'
    # ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact']
    t0 = time()
    res_m = minimize(male_process,x0=x0, method=opt_method)
    t1 = time()
    #res_f = minimize(male_process, optimizer_method=opt_method)
    #t2 = time()


'''
