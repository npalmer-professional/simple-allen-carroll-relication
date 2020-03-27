Simple Allen Carroll Replication
================================

This repository contains minimal code that can be used to replicate [Allen and Carroll (2001)](http://www.econ2.jhu.edu/people/ccarroll/IndivLearningAboutC.pdf). 

An extension: get two versions of this code working: 

1. fast social learning
2. fast learning with portfolio choice

Chris Carroll References for learning:

http://www.econ2.jhu.edu/people/ccarroll/courses/Choice/LectureNotes/Consumption/
http://www.econ2.jhu.edu/people/ccarroll/courses/choice/LectureNotes/
http://www.econ2.jhu.edu/people/ccarroll/teaching.html#Choice
http://www.econ2.jhu.edu/people/ccarroll/courses/Topics/Syllabus.html


Working out Perpetual Youth Combinatorics
-----------------------------------------


Assume life: 3 periods
Prob die: two states:  p1, p2
prob transition from one state to another: q1

Sequence:

- wake
- flip coin to die
- if live, flip coin to transition
- sleep

So here's the coin flips:

Day1:

p_alive in morning = 1
p_in_state_1 = 1
- flip coin to live/die
- if live, flip coin to transition
p_alive in evening = (1-p1)
p_in_state_1 = (1-p1)*(1-q1)
- sleep
- wake
p_alive_in_morning = (1-p1)
- flip coin to live/die:
    - if *didn't* transition last night, use p1
    - if transitioned last night, use p2
p_alive in evening = (1-p1)*             // lived last night, and 
                         ((1-q1)*(1-p1)  // did't transition and didn't die, *or*
                         + q1*(1-p2))    // transitioned and didn't die
- if live, flip coin to transition:
    - prob still in state 1: (1-q1)*(1-q1)
    - prob in state 2:       (1-q1)* q1
- sleep
- wake
p_alive in morning = (1-p1)*             // lived last night, and 
                         ((1-q1)*(1-p1)  // did't transition and didn't die, *or*
                         + q1*(1-p2))    // transitioned and didn't die
- flip coin to see if live
    - let's figure out the ways could get to today:
        1. (1-p1) *          // lived first day, and 
        2. ((1-q1)*(1-p1)    // didn't transition and didn't die, or
           + q1 * (1-p2)     // transitioned and didn't die


















