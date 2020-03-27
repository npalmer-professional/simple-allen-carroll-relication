'''
Very basic perfect foresight consumer.

Following: http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA/

Goal: direct solution and fast code. 
'''

import numpy as np


class PerfectForesightAgent(object):
    def __init__(self, β, ρ):
        '''Initialize a perfect foresight agent.
        Parameters
        β : float
            Time preference
        ρ : float
            Risk aversion
        '''
        self.β, self.ρ = β, ρ


    def þ(self, R):
        '''Calculate the absolute patience factor þ ≡ (Rβ)**(1/ρ)

        This is the RHS of the standard consumption Euler:
            c_{t+1} / c_t  =  (Rβ)**(1/ρ)

        When þ < 1 → "absolute impatience" because c_t guaranteed to fall
        When þ > 1 → "absolute patience" because c_t guaranteed to grow 
        
        Note that this is a property of preference parameters and the rate of 
        return. ("Percieved rate of return.")

        Parameters
        R : float   -- Return factor from period t to t+1
        Returns
        þ : float   -- The absolute patience factor
        '''
        return (R*self.β)**(1/self.ρ)


    def euler_error(self, c_t, c_tp1, R):
        '''  Calculate the Euler error given consumption choices, return R.
        Recall that þ is the RHS of the typical consumption Euler.
        Parameters
        c_t, c_tp1 : float
            Consumption in periods t and t+1
        R : float
            Return factor from period t to t+1
        '''
        return c_tp1/c_t - self.þ

    def ℙ(self, x_t, X, t, T, R):
        '''  Calculate the present discounted value of observation x.
        For constant-growth-factor values (eg. c_t, p_t). 
    
        To get PDV of consumption, set:
            x_t ≡ c_t  # consumption
            X   ≡ þ    # growth factor for consumption

        To get PDV of labor income (human wealth), set:
            x_t ≡ p_t  # permanent income
            X   ≡ G    # growth factor of permanent income

        Parameters
        x_t : float
            Consumption (permanent income) in period t
        X : float
            Constant growth factor of consumption (permanent income) 
        t, T : int
            Current and final time period, respectively
        R : float
            Return factor from period t to t+1
        '''
        return x_t * ( (1-(X/R)**(T-t+1)) / (1-(X/R)) )
    
    def IBC_error(self, b_t, c_t, p_t, t, T, G, R):
        '''Intertemporal budget constraint error. 
        Difference between PDV consumption and PDV_labor + bank balances.
        Comes from the intertemporal budget constraint:
            ℙ(c) = b_t + ℙ(p)
        
        Error should be zero -- if positive, more wealth than 
        consumption; if negative, then more consumption than 
        wealth.
        '''
        PDV_c = self.ℙ(c_t, self.þ(R), t, T, R) # PDV consumption
        PDV_h = self.ℙ(p_t, G, t, T, R)         # PDV labor: "human wealth"

        # PDV total resources - PDV consumption
        return (b_t + PDV_h) - PDV_c
    
    def κ(self, t, T R):
        '''Kappa, the marginal propensity to consume out of total wealth. 
        This comes algebraically solving out the IBC.
        '''
        þ = self.þ(R)
        return (1 - þ/R) / (1 - (þ/R)**(T-t+1) )

    def o(self, b_t, p_t, t, T, G, R):
        '''Calculate overall wealth: human welath + nonhuman wealth.
        '''
        return b_t + self.ℙ(p_t, G, t, T, R)

    def c(self, b_t, p_t, t, T, G, R):
        return self.κ(t, T R) * self.o(b_t, p_t, t, T, G, R)

    def finite_human_wealth_condition(self, G, R):
        '''Return True if G, R fulfils the finite human wealth condition.
        For the infnite horizon case, where T → ∞ 
        '''
        return G < R

    def þ_R(self, R):
        '''Return True if G, R fulfils the finite human wealth condition.
        For the infnite horizon case, where T → ∞ 
        '''
        return self.þ(R) / R < 1

