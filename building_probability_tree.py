# Let's try to build out the probability tree that need to populate to 
# calculate the probability associated with the multi-stage perpetual youth 
# model

# build transition probabilities: 
pλ = [0.99,  0.975] # probability that live -- i.e. don't die. λ for "live"
qσ = [0.075, 0.0]   # probability that don't transition -- i.e. stay the same.
                    # σ for "stay"  

max_i = len(pλ) - 1  # highest i-value
tree = {}
age = 5


# Enter the first period in the tree:
tree[(0,)] = {'state':0,
             'prob': 1.0,
             'visited':False,
             'left node id':None,
             'right node id':None}
'''
Let's build out a tree to age 1, then 2, then 3, then maybe 4

The upper-most path is the "no death, no change"

(tm) -> time period, morning
(te) -> time period, evening
                                                  states to get here:
(0m) p0 (1)   q0 p0 (2)   q0 p0 (3)   q0 p0 (4)   0000
                                    1-q0 p1 (4)   0001
                        1-q0 p1 (3)   q1 p1 (4)   0011
            1-q0 p1 (2)   q1 p1 (3)   q1 p1 (4)   0111
                        1-q1 NA

So let's buid the prob that get to each age:

1:  p0
2:  p0 * (q0*p0) + p0*(1-q0)*p1   =   p0*(q0*p0 + (1-q0)*p1)
3:  p0 * q0*p0 * q0*p0  +  p0 * q0*p0 * (1-q0)*p1  +  p0 * (1-q0)*p1 * p1
4:  p0*    q0*p0*    q0*p0*    q0*p0 + 
    p0*    q0*p0*    q0*p0*(1-q0)*p1 + 
    p0*    q0*p0*(1-q0)*p1*       p1 + 
    p0*(1-q0)*p1*       p1*       p1


Now let's build out the 5th period just to see:


                                                        states to get here:
(0m) p0 (1)   q0 p0 (2)   q0 p0 (3)   q0 p0 (4)   q0 p0 (5)     00000
                                                1-q1 p1 (5)     00001
                                    1-q0 p1 (4)      p1 (5)     00011
                        1-q0 p1 (3)      p1 (4)      p1 (5)     00111
            1-q0 p1 (2)      p1 (3)      p1 (4)      p1 (5)     01111

States that get to each stage:
t1      t2      t3      t4      t5
0       00      000     0000    00000
                                00001
                        0001    00011
                001     0011    00111
        01      011     0111    01111


Now let's build out a three-state tree:
pλ = [0.99,  0.975, 0.95]
qσ = [0.075, 0.10,  0.0]

(0) p0 (1)   q0 p0 (2)   q0 p0 (3)   q0 p0 (4)   q0 p0 (5)  00000  t=5,
                                               1-q0 p1 (5)  00001  t
                                   1-q0 p1 (4)   q1 p1 (5)  00011
                                               1-q1 p2 (5)  00012
                       1-q0 p1 (3)   q1 p1 (4)   q1 p1 (5)  00111
                                               1-q1 p2 (5)  00112
                                   1-q1 p2 (4)      p2 (5)  00122
           1-q0 p1 (2)   q1 p1 (3)   q1 p1 (4)   q1 p1 (5)  01111
                                               1-q1 p2 (5)  01112
                                   1-q1 p2 (4)      p2 (5)  01122
                       1-q1 p2 (3)      p2 (4)      p2 (5)  01222


BUILD THE TREE: given an age, do what I did above. At each age, attach all 
branches. Binary s should be fine. Keep track of a "master age list"; 
at each age in the master age list, keep a list that points to all nodes at 
this age. When building the list, step forward one age at a time. 


'''
# First lets just walk all the way down the left side -- this is the 
# lowest-state side:
t = 0
i = 0
current_node_id = (0,)
while t <= age:

    # Create ID of next node:
    left_node_id = tuple([a for a in current_node_id] + [i])
        # Note: left node is always the same as the current node
    if i+1 
    right_node_id = tuple([a for a in current_node_id] + [i+1])

    # Fill in the current 

    # check if can go left -- should be able to at this point:
    #if i <= max_i:
        # then can go left
        








