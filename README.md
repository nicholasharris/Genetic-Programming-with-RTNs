# Genetic-Programming-with-RTNs
A python implementation of genetic programming, using the turing-complete Recursive Tree Network (RTN) structure defined by Taro Yabuki and Hitoshi Iba in the paper "Genetic Programming Using a Turing-Complete Representation: Recurrent Network Consisting of Trees". Link: https://www.researchgate.net/publication/250859796_Genetic_Programming_Using_a_Turing-Complete_Representation_Recurrent_Network_Consisting_of_Trees

Genetic programming uses a genetic algorithm over the space of possible programs to produce a program that performs a desired task.
The traditional way of doing this uses program trees - however this representation is not Turing-complete, so there are many programs it cannot represent. The RTN structure defined in the paper I name above is turing complete, and theoretically any program can be represented with an RTN.

Provided are the libarary file "genetic_programming.py", which contains the implementation of RTNs and a population class to easily run a Genetic Algorithm with the RTNs.
In addition, I include an example test file, "gp_fibonacci_test.py", which evolves a program to produce the fibonacci sequence. Using the paramaters and random seed included in that file, a correct fibonacci generator is evolved in generation 8.

I arbitrarily set the size limit of numbers the trees will handle at [-1000, 1000], this can be easily modified to suit a different problem domain if desired.

Tested in Python 3.6 and Windows 10. 
