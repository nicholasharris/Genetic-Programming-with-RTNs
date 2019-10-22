##################################
# An implementation of turing-complete genetic programming with
#   RTN's (Recursive Tree Networks) in Python 3
#   
#
# As outlined in the paper: http://www.unfindable.net/paper/2003-yabuki-rtn-draft.pdf
#
# Nicholas Harris
##################################


import random
import math
import numpy
from functools import reduce
from collections import Counter

#RTN population - the highest level object for us in genetic algorithm
class RTN_population:
    def __init__(self, pop_size, max_nodes, max_tree_depth, num_inputs, elitism, diversity_generate, fitness_sharing, preserve_diversity):
        self.RTNs = []
        self.idCounter = 0
        self.elitism = elitism
        self.diversity_generate = diversity_generate
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.max_tree_depth = max_tree_depth
        self.gen = 0
        self.fitness_sharing = fitness_sharing
        self.preserve_diversity = preserve_diversity
        self.num_inputs = num_inputs

        for x in range(pop_size):
            chrom = RTN(random.randint(1,max_nodes), num_inputs, random.randint(1, max_tree_depth))
            chrom.ID = self.idCounter
            self.idCounter += 1
            self.RTNs.append(chrom)

    #To produce new population via selection, crossover, and mutation
    def eval_genomes(self):
        print("\n\n****** GP PYTHON ||||| GENERATION STEP " + str(self.gen) + " ********\n")
        fitnesses = []
        fitnessTotal = 0.0
        fitness_pairs = []
        for x in range(self.pop_size):
            fitnesses.append(self.RTNs[x].fitness)
            fitnessTotal += self.RTNs[x].fitness
            fitness_pairs.append( [self.RTNs[x], fitnesses[x]] )


        normalized_fitnesses = []
        normalized_fitness_pairs = []
        normalized_fitness_total = 0
        if self.fitness_sharing == True:    #If fitness sharing is on, normalize the genome's fitnesses
            counted_fitnesses = Counter(fitnesses)
            for x in range(self.pop_size):
                normalized_fitnesses.append( fitnesses[x]/counted_fitnesses[fitnesses[x]] )
                normalized_fitness_pairs.append( [self.RTNs[x], normalized_fitnesses[x] ])
                normalized_fitness_total += normalized_fitnesses[x]
                self.RTNs[x].normalized_fitness = normalized_fitnesses[x]

        fitnesses = list(reversed(sorted(fitnesses))) #fitnesses now in descending order
        sorted_pairs = list(reversed(sorted(fitness_pairs, key=lambda x: x[1])))

        

        new_RTNs = []
        new_RTNs_fitnesses = []
        #preserve some RTNs through elitism
        if self.preserve_diversity == False:
            for x in range(self.elitism):
                new_RTNs.append(sorted_pairs[x][0])
        else:
            count = 0
            while len(new_RTNs) < self.elitism and count < self.pop_size:
                if fitnesses[count] not in set(new_RTNs_fitnesses):
                    new_RTNs.append(sorted_pairs[count][0])
                    new_RTNs_fitnesses.append(fitnesses[count])
                count += 1
                

        best_RTN = sorted_pairs[0][0]
        

        if (self.fitness_sharing == False):
            print("   Best RTN ||  Fitness: " + str(best_RTN.fitness) + "\n  ||  # nodes: " + str((best_RTN.size)) + "  ||  Tree Depth: " + str(best_RTN.depth) + " || ID: " + str(best_RTN.ID) + "\n")
        else:
            print("   Best RTN ||  Fitness: " + str(best_RTN.fitness) + "  ||  Normalized Fitness: " + str(best_RTN.normalized_fitness) + "\n  ||  # nodes: " + str((best_RTN.size)) + "  ||  Tree Depth: " + str(best_RTN.depth) + " || ID: " + str(best_RTN.ID) + "\n")

        
        average_fitness = reduce(lambda x, y: x + y, fitnesses)/float(len(fitnesses))
        print("   Average Fitness: " + str(average_fitness) + "\n")

        #If using fitness sharing, replace the notion of fitness now with the normalized fitness, after elitism has been preserved.
        if (self.fitness_sharing == True):
            fitnesses = list(reversed(sorted(normalized_fitnesses)))
            sorted_pairs = list(reversed(sorted(normalized_fitness_pairs, key=lambda x: x[1])))
            fitnessTotal = normalized_fitness_total

        #create roulette wheel from relative fitnesses for fitness proportional selection
        rouletteWheel = []
        fitnessProportions = []
        for i in range(self.pop_size):
            fitnessProportions.append( float( fitnesses[i]/fitnessTotal ) )
            if(i == 0):
                rouletteWheel.append(fitnessProportions[i])
            else:
                rouletteWheel.append(rouletteWheel[i - 1] + fitnessProportions[i])
    
        #Generate most new population with children of selected RTNs
        while len(new_RTNs) < self.pop_size - self.diversity_generate:

            #Fitness Proportional Selection
            spin1 = random.uniform(0, 1)      # A random float from 0.0 to 1.0
            spin2 = random.uniform(0, 1)      # A random float from 0.0 to 1.0

            j = 0
            while( rouletteWheel[j] <= spin1 ):
                j += 1

            k = 0
            while( rouletteWheel[k] <= spin2 ):
                k += 1

                
            parent1 = RTN(sorted_pairs[j][0].size, sorted_pairs[j][0].num_inputs, sorted_pairs[j][0].depth)   #parent 1

            parent1.topology_vector = []
            for x in range(parent1.size):
                vector = []
                for y in range(parent1.num_inputs):
                    vector.append(sorted_pairs[j][0].topology_vector[x][y])
                parent1.topology_vector.append(vector)

            for x in range(parent1.size):
                copy_into(sorted_pairs[j][0].trees[x].root, parent1.trees[x].root)
            
            parent2 = RTN(sorted_pairs[k][0].size, sorted_pairs[k][0].num_inputs, sorted_pairs[k][0].depth)  #parent 2

            parent2.topology_vector = []
            for x in range(parent2.size):
                vector = []
                for y in range(parent2.num_inputs):
                    vector.append(sorted_pairs[k][0].topology_vector[x][y])
                parent2.topology_vector.append(vector)

            for x in range(parent2.size):
                copy_into(sorted_pairs[k][0].trees[x].root, parent2.trees[x].root)
            

            #create child from parents (Crossover)

            # Tree crossover (actually modifies the program trees via crossover)
            if (random.random() < 0.5):
                for x in range(min(parent1.size, parent2.size)):
                    #select trees to cross
                    index1 = random.randint(0, parent1.size - 1)
                    index2 = random.randint(0, parent2.size - 1)

                    #select nodes to cross (segments of tree to transplant)
                    node1 = leaf()
                    while( node1.left_child is None):
                        node = parent1.trees[index1].root
                        while(node.left_child is not None):
                            if random.random() < 0.2:
                                node1 = node
                            if random.random() < 0.5:
                                node = node.left_child
                            else:
                                node = node.right_child

                    node2 = leaf()
                    while( node2.left_child is None):
                        node = parent2.trees[index2].root
                        while(node.left_child is not None):
                            if random.random() < 0.2:
                                node2 = node
                            if random.random() < 0.5:
                                node = node.left_child
                            else:
                                node = node.right_child

                    temp = leaf()
                    temp.value = node1.value
                    temp.type = node1.type
                    temp.single_op = node1.single_op
                    temp.left_child = node1.left_child
                    temp.right_child = node1.right_child

                    node1.value = node2.value
                    node1.type = node2.type
                    node1.single_op = node2.single_op
                    node1.left_child = node2.left_child
                    node1.right_child = node2.right_child

                    node2.value = temp.value
                    node2.type = temp.type
                    node2.single_op = temp.single_op
                    node2.left_child = temp.left_child
                    node2.right_child = temp.right_child

                    
            # Tree exchange (swap trees as-is)
            if (random.random() < 0.5):
                for x in range(min(parent1.size, parent2.size)):
                    index1 = random.randint(0, parent1.size - 1)
                    index2 = random.randint(0, parent2.size - 1)

                    temp = leaf()
                    temp.value = parent1.trees[index1].root.value
                    temp.type =  parent1.trees[index1].root.type
                    temp.single_op =  parent1.trees[index1].root.single_op
                    temp.left_child =  parent1.trees[index1].root.left_child
                    temp.right_child =  parent1.trees[index1].root.right_child

                    parent1.trees[index1].root.value = parent2.trees[index2].root.value
                    parent1.trees[index1].root.type = parent2.trees[index2].root.type
                    parent1.trees[index1].root.single_op = parent2.trees[index2].root.single_op
                    parent1.trees[index1].root.left_child = parent2.trees[index2].root.left_child
                    parent1.trees[index1].root.right_child = parent2.trees[index2].root.right_child

                    parent2.trees[index2].root.value = temp.value
                    parent2.trees[index2].root.type = temp.type
                    parent2.trees[index2].root.single_op = temp.single_op
                    parent2.trees[index2].root.left_child = temp.left_child
                    parent2.trees[index2].root.right_child = temp.right_child

            #Topology exchange
            if (random.random() < 0.5):
                for x in range(min(parent1.size,parent2.size)):
                    temp = []
                    index1 = random.randint(0, parent1.size - 1)
                    index2 = random.randint(0, parent2.size - 1)
                    for x in range(parent1.num_inputs):
                        temp.append(parent1.topology_vector[index1][x])

                    for x in range(parent1.num_inputs):
                        parent1.topology_vector[index1][x] = parent2.topology_vector[index2][x] % parent1.size

                    for x in range(parent2.num_inputs):
                        parent2.topology_vector[index2][x] = parent1.topology_vector[index1][x] % parent2.size
            #Node exchange 
            if (random.random() < 0.5):
                index1 = random.randint(0, parent1.size - 1)
                index2 = random.randint(0, parent2.size - 1)

                temp = leaf()
                temp.value = parent1.trees[index1].root.value
                temp.type =  parent1.trees[index1].root.type
                temp.single_op =  parent1.trees[index1].root.single_op
                temp.left_child =  parent1.trees[index1].root.left_child
                temp.right_child =  parent1.trees[index1].root.right_child

                parent1.trees[index1].root.value = parent2.trees[index2].root.value
                parent1.trees[index1].root.type = parent2.trees[index2].root.type
                parent1.trees[index1].root.single_op = parent2.trees[index2].root.single_op
                parent1.trees[index1].root.left_child = parent2.trees[index2].root.left_child
                parent1.trees[index1].root.right_child = parent2.trees[index2].root.right_child

                parent2.trees[index2].root.value = temp.value
                parent2.trees[index2].root.type = temp.type
                parent2.trees[index2].root.single_op = temp.single_op
                parent2.trees[index2].root.left_child = temp.left_child
                parent2.trees[index2].root.right_child = temp.right_child

                temp = []
                for x in range(parent1.num_inputs):
                    temp.append(parent1.topology_vector[index1][x])

                for x in range(parent1.num_inputs):
                    parent1.topology_vector[index1][x] = parent2.topology_vector[index2][x] % parent1.size

                for x in range(parent2.num_inputs):
                    parent2.topology_vector[index2][x] = parent1.topology_vector[index1][x] % parent2.size
                
                        
                    
            #mutate genome
            child = parent1
            if random.random() < 0.5:
                child = parent2

            #Topology mutation
            if random.random() < 0.05:
                index = random.randint(0, child.size - 1)
                child.trees[index] = program_tree(random.randint(1, self.max_tree_depth), self.num_inputs)

            #Tree mutation
            if random.random() < 0.05:
                index = random.randint(0, child.size - 1)
                child.topology_vector[index] = []
                for x in range(child.num_inputs):
                    child.topology_vector[index].append(random.randint(0, child.size - 1))

            #Add new RTN to popultaion
            child.ID = self.idCounter
            self.idCounter += 1
            new_RTNs.append(child)
           
        #Generate a number of random individuals for diversity in the population
        for x in range(self.diversity_generate):
            new = RTN(random.randint(1,self.max_nodes), self.num_inputs, random.randint(1, self.max_tree_depth))
            new_RTNs.append(new)
            

        self.RTNs = new_RTNs
        random.shuffle(self.RTNs)

        for x in range(self.pop_size):
            self.RTNs[x].reset()
            self.RTNs[x].fitness = 0
            
        self.gen = self.gen + 1
        print("\n*************** GENERATION STEP FINISH ********************\n\n")
        
#Recursive Tree Network
class RTN:

    def __init__(self, size, num_inputs, tree_depth):
        self.size = size
        self.num_inputs = num_inputs
        self.trees = []
        self.time_step = 0
        self.fitness = 0
        self.normalized_fitness = 0
        self.depth = tree_depth
        self.ID = 0
        
        #Program trees or s-expressions
        for x in range(size):
            self.trees.append(program_tree(tree_depth, num_inputs))

        #Network topology
        self.topology_vector = []
        for x in range(size):
            vector = []
            for y in range(num_inputs):
                vector.append(random.randint(0, self.size - 1))
            self.topology_vector.append(vector)

        #Initial State
        self.state = []
        for x in range(size):
            self.state.append(1)

    def reset(self):
        #Initial State
        self.state = []
        for x in range(self.size):
            self.state.append(1)

        self.time_step = 0

    def tick(self, inputs):
        outputs = []
        for x in range(self.size):
            outputs.append(self.trees[x].activate(self.trees[x].root, inputs, self.time_step, self.state, self.topology_vector[x]))

        for x in range(self.size):
            if (outputs[x] > 1000):
                outputs[x] = 1000
            elif outputs[x] < -1000:
                outputs[x] = -1000
            self.state[x] = outputs[x]

        self.time_step += 1

#Program tree or S-expression
class program_tree:

    def __init__(self, depth, num_inputs):
        self.depth = depth
        self.root = leaf()

        self.initialize_tree_topology(0, self.root)
        self.initialize_tree_values(self.root, num_inputs)

    def activate(self, node, inputs, time_step, old_state, topology_vector):

        if node.type == "const":
            if node.single_op is not None:
                return single_operation(node.single_op, node.value)
            else:
                return node.value

        elif node.type == "var":
            if time_step == 0:
                if node.single_op is not None:
                    return single_operation(node.single_op, inputs[node.value])
                else:
                    return inputs[node.value]
            else:
                if node.single_op is not None:
                    return single_operation(node.single_op, old_state[topology_vector[node.value]])
                else:
                    return old_state[topology_vector[node.value]]

        else:    #node is an operation

            #Recursively activate tree
            val = operation(node.value, self.activate(node.left_child, inputs, time_step, old_state, topology_vector), self.activate(node.right_child, inputs, time_step, old_state, topology_vector))

            if val > 0:
                return min(val ,1000)
            else:
                return max(val, -1000)
        

    def initialize_tree_topology(self, my_depth, node):

        if(my_depth < self.depth):

            if random.random() < 1.0 - ((1/self.depth)*my_depth)/2.0:
                node.left_child = leaf()
                self.initialize_tree_topology(my_depth + 1, node.left_child)
        
                node.right_child = leaf()
                self.initialize_tree_topology(my_depth + 1, node.right_child)

    def initialize_tree_values(self, node, num_inputs):

        if node.left_child is not None and node.right_child is not None:
            node.value = random.randint(0, len(op_switcher) - 1)
            node.type = "op"

            self.initialize_tree_values(node.left_child, num_inputs)
            self.initialize_tree_values(node.right_child, num_inputs)

            if random.random() < 0.15:
                node.single_op = random.randint(0, len(sop_switcher) - 1)

        else:
            if random.random() < 0.5:
                node.type = "var"
                node.value = random.randint(0, num_inputs - 1)
            else:
                node.type = "const"
                node.value = random.random()*20 - 10

            if random.random() < 0.15:
                node.single_op = random.randint(0, len(sop_switcher) - 1)

#function to copy tree into another tree structure
def copy_into(node, new_node):
    new_node.value = node.value
    new_node.type = node.type
    new_node.single_op = node.single_op
    if node.left_child is not None:
        new_node.left_child = leaf()
        copy_into(node.left_child, new_node.left_child)
    if node.right_child is not None:
        new_node.right_child = leaf()
        copy_into(node.right_child, new_node.right_child)
            
#Leaf of program tree
class leaf:

    def __init__(self):
        self.value = None
        self.type = None
        self.single_op = None
        self.left_child = None
        self.right_child = None

#Define component operations for program tree
def addition(a, b):
    return a + b

def multiplication(a, b):
    return a * b

def subtraction(a, b):
    return a - b

def division(a, b):
    if b == 0:
        return 1
    else:
        return a/b

def mod(a, b):
    if b < 1:
        return 1
    else:
        return int(a) % int(b)

def exponent(a, b):
    if(int(abs(a)) == 0 and int(abs(b)) == 0):
        return 1
    return int(abs(a))**int(abs(b))


def logarithm(a, b):

    if b < 0.001 or a < 0.001 or (b > 0.999 and b < 1.001):
        return 1
    else:
        #print("a, b: " + str(a)  + " , " + str(b))
        return math.log(a, b)

def maximum(a, b):
    return max(a, b)

def minimum(a, b):
    return min(a, b)

def mean(a, b):
    return (a + b)/2.0

def p(a):
    return a % 2

def h(a):
    return int(a/2)

def ln(a):
    if (a < 1.001):
        return 1
    return math.log(a)

def inverse(a):
    if (a < 0.001 and a > -0.001):
        return 1
    else:
        return 1/a

def floor(a):
    return int(a)

def ceil(a):
    return math.ceil(a)

#Dictionary of operations with 2 arguments
#   To be used as a switch
op_switcher = {
        0: addition,
        1: multiplication,
        2: subtraction,
        3: division,
        4: mod,
        5: exponent,
        6: logarithm,
        7: maximum,
        8: minimum,
        9: mean 
    }

#Dictionary of operations with 1 argument
#   To be used as a switch
sop_switcher = {
        0: p,
        1: h,
        2: ln,
        3: inverse,
        4: floor,
        5: ceil
    }

#Collection of arithmetic operations, selectable by index
def operation(n, a, b):

    op = op_switcher.get(n, lambda: "Invalid Function Index")

    #Execute the selected operation
    return op(a, b)

#Collection of single operations (one input)
def single_operation(n,a):
    sop = sop_switcher.get(n, lambda: "Invalid single-op Index")

    #Execute operation
    return sop(a)

        
