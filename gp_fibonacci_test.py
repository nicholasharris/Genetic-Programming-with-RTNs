##################################
# An implementation of turing-complete genetic programming with
#   RTN's (Recursive Tree Networks) in Python 3
# As outlined in the paper "Genetic Programming Using a Turing-Complete Representation:
#       Recurrent Network Consisting of Trees" by Taro Yabuki and Hitoshi Iba.
#
# This test evolves a population of RTNs to produce a fibonacci generator
#
# Nicholas Harris
##################################


import random
import math
import numpy
import genetic_programming as gp

#Implementation of Fibonacci to test RTNs against
def F(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return F(n-1)+F(n-2)

#Arguments:
#Pop size, max RTN size, max tree depth, num inputs, num elitism, num diversity generate, fitness sharing(bool), preserve diversity(bool)
my_RTNs = gp.RTN_population(5000, 4, 3, 4, 100, 1000, True, True)

NUM_GENS = 1000

# The above population settings with this seed results in a correct fibonacci generator being created
#     in generation 132.
random.seed(456845684568)

vals = []
for y in range(17):
    vals.append(F(y))

for x in range(NUM_GENS):


    for y in range(10):
        for z in range(my_RTNs.pop_size):
            my_RTNs.RTNs[z].tick([1,1,1,1])
            if my_RTNs.RTNs[z].state[0] == vals[y]:
                my_RTNs.RTNs[z].fitness += 1

    best_fitness = 0
    best_id = 0
    for y in range(my_RTNs.pop_size):
        if my_RTNs.RTNs[y].fitness > best_fitness:
            best_fitness = my_RTNs.RTNs[y].fitness
            best_id = my_RTNs.RTNs[y].ID
        if my_RTNs.RTNs[y].fitness == 10:
            my_RTNs.RTNs[y].reset()
            print("Fitness 10 reached. Trying generalization test...")
            vals = []
            for z in range(17):
                vals.append(F(z))

            score = 0
            for z in range(17):
                my_RTNs.RTNs[y].tick([1,1,1,1])
                print( "real, observed: " + str(vals[z]) + " , " + str(my_RTNs.RTNs[y].state[0]))
                if my_RTNs.RTNs[y].state[0] == vals[z]:
                    score += 1

            print("Score: " + str(score))

            break

    if x % 20 == 0:
        for y in range(len(my_RTNs.RTNs)):
            if my_RTNs.RTNs[y].ID == best_id:
                my_RTNs.RTNs[y].reset()
                print("Best ID: " + str(best_id))
                print("Trying big test on best chromosome...")
                
                score = 0
                for z in range(17):
                    my_RTNs.RTNs[y].tick([1,1,1,1])
                    print( "real, observed: " + str(vals[z]) + " , " + str(my_RTNs.RTNs[y].state[0]))
                    if my_RTNs.RTNs[y].state[0] == vals[z]:
                        score += 1

                print("Score: " + str(score))

    my_RTNs.eval_genomes()


#print("resetting...")
#my_RTN.reset()

#my_RTN.tick([1, 2, 3, 4 ,5])
#print( "state:" + str(my_RTN.state))
 

 

        
