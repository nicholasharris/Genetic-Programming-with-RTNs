# Genetic-Programming-with-RTNs
A python implementation of genetic programming, using the turing-complete Recursive Tree Network (RTN) structure defined by Taro Yabuki and Hitoshi Iba in the paper "Genetic Programming Using a Turing-Complete Representation: Recurrent Network Consisting of Trees".  

Provided are the libarary file "genetic_programming.py", which contains the implementation of RTNs and a population class to easily run a Genetic Algorithm with the RTNs.
In addition, I include an example test file, "gp_fibonacci_test.py", which evolves a program to produce the fibonacci sequence. Using the paramaters included in that file, a correct fibonacci generator is evolved in generation 132 (this takes only a couple minutes on my computer).
