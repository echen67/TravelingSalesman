import random
import sys
import argparse
from timeit import default_timer as timer
import numpy as np
import math

# Global variable
BnBbestQuality = float('inf')
BnBbestTour = []
BnBtrace = []
visited = []


# Branch and Bound algorithm
def BnB(nodes, time):
    '''
    Args:
    -    nodes: N x 3 array of nodes
    -    time: cut-off time in seconds

    Returns:
    -    quality: quality of best solution found
    -    tour: list of node IDs
    -    trace: list of best found solution at that point in time. Record every time a new improved solution is found.
    '''
    global BnBbestTour
    global visited

    start = timer()
    BnBbestTour = [None] * (len(nodes) + 1)
    visited = [None] * (len(nodes))

    for i in range(len(BnBbestTour)):
        BnBbestTour[i] = -1
    for y in range(len(visited)):
        visited[y] = False

    lb = 0
    for x in nodes:
        lb += (minimum(x, nodes) + second_minimum(x, nodes)) / 2

    # Initialize the start point
    randNum = random.randint(0, len(nodes))
    visited[randNum-1] = True
    BnBbestTour[0] = randNum-1

    BnBHelp(nodes, start, 1, lb, 0, float(time))
    return BnBbestQuality, BnBbestTour, BnBtrace


def BnBHelp(nodes, time, level, lb, weight, timeLimit):
    global BnBbestQuality
    global BnBbestTour
    global BnBtrace
    global visited

    # Finalize values after all nodes are traversed
    if level == len(nodes):
        newQuality = weight + distance(nodes[BnBbestTour[level - 1]][1], nodes[BnBbestTour[level - 1]][2],
                                       nodes[BnBbestTour[0]][1], nodes[BnBbestTour[0]][2])
        print(newQuality)
        print(BnBtrace)
        print(BnBbestTour, '\n')
        if newQuality < BnBbestQuality:
            BnBbestTour[len(nodes)] = BnBbestTour[0]
            BnBbestQuality = newQuality
            BnBtrace.append([timer() - time, BnBbestQuality])

            print(newQuality)
            print(BnBtrace)
            print(BnBbestTour, '\n')
            return

    # If there are still nodes not traversed, check whether we should continue iterating further
    for i in range(len(nodes)):
        # print(level)
        # print(i)
        if timer() - time > timeLimit:
            print("Time Limit met")
            break

        if not visited[i]:
            temp_lb = lb
            weight += distance(nodes[BnBbestTour[level - 1]][1], nodes[BnBbestTour[level - 1]][2],
                               nodes[i][1], nodes[i][2])
            if level == 1:
                lb -= (minimum(nodes[BnBbestTour[level - 1]], nodes) + minimum(nodes[i], nodes)) / 2
            else:
                lb -= (second_minimum(nodes[BnBbestTour[level - 1]], nodes) + minimum(nodes[i], nodes)) / 2

            # Calculate the lower bound for the current node
            if lb + weight < BnBbestQuality:
                BnBbestTour[level] = int(nodes[i][0])-1
                visited[i] = True
                BnBHelp(nodes, time, level + 1, lb, weight, timeLimit)
            else:
                weight -= distance(nodes[BnBbestTour[level - 1]][1], nodes[BnBbestTour[level - 1]][2],
                                   nodes[i][1], nodes[i][2])
                lb = temp_lb
                for y in range(len(visited)):
                    visited[y] = False
                for j in range(level):
                    visited[BnBbestTour[j] - 1] = True


def minimum(node, nodes):
    curr_min = float('inf')
    for x in nodes:
        curr_dist = distance(x[1], x[2], node[1], node[2])
        if curr_dist < curr_min and x != node:
            curr_min = curr_dist
    return curr_min


def second_minimum(node, nodes):
    first_min, second_min = float('inf'), float('inf')
    for x in nodes:
        curr_dist = distance(x[1], x[2], node[1], node[2])
        if curr_dist <= first_min:
            second_min = first_min
            first_min = curr_dist
        elif curr_dist < second_min:
            second_min = curr_dist
    return second_min


def Approx(nodes, time, seed):
    '''
    Args:
    -    nodes: N x 3 array of nodes
    -    time: cut-off time in seconds
    -    seed: random seed

    Returns:
    -    quality: quality of best solution found
    -    tour: list of node IDs
    -    trace: list of best found solution at that point in time. Record every time a new improved solution is found.
    '''
    # Dummy values
    quality = 0
    tour = [1, 2, 3]
    trace = [[3.45, 102], [7.94, 95]]
    return quality, tour, trace


def LS1(nodes, time, seed):
    '''
    Simulated annealing.

    Args:
    -    nodes: N x 3 array of nodes
    -    time: cut-off time in seconds
    -    seed: random seed

    Returns:
    -    quality: quality of best solution found
    -    tour: list of node IDs
    -    trace: list of best found solution at that point in time. Record every time a new improved solution is found.
    '''

    # STILL NEED TO OPTIMIZE - USE TIME LIMIT?
    # Better stopping criterion - no improved solution for a certain number of iterations
    # Restarts?
    # ADD TABU???

    start = timer()
    trace = []
    coolingRate = 0.0005
    temperature = 10000

    # Use the seed to get a random initial tour
    print("nodes: ", len(nodes))
    np.random.seed(seed)
    currentTour = np.random.permutation(np.arange(len(nodes)))  # contains INDEX of nodes
    print("currentTour: ", currentTour)

    # Evaluate distance of this random initial tour
    currentQuality = evaluation_function(currentTour, nodes)
    print("currentQuality: ", currentQuality)

    # Keep track of best solution
    bestTour = currentTour
    bestQuality = currentQuality

    # np.random.seed()	# Reset seed, otherwise you'll get exact same final answer every time. Or that what we want???

    # Consider a different stopping condition - iteration not based on temp, bounded by time
    while (temperature > 0.99):
        # Stop if time has exceeded limit
        end = timer()
        if end >= time:
            print("TIME LIMIT")
            break

        # Create new tour. 2-opt exchange. Do you just randomly get four points? Is that considered adjacent?
        randPoints = np.random.choice(np.arange(len(currentTour)), size=4, replace=False)
        newTour = currentTour.copy()
        save2 = newTour[randPoints[1]]
        newTour[randPoints[1]] = newTour[randPoints[0]]
        newTour[randPoints[0]] = save2
        save4 = newTour[randPoints[3]]
        newTour[randPoints[3]] = newTour[randPoints[2]]
        newTour[randPoints[2]] = save4
        newQuality = evaluation_function(newTour, nodes)

        # Calculate probability based on temperature
        probability = np.exp(-(newQuality - currentQuality) / temperature)  # FIX OVERFLOW WARNING
        randNum = np.random.random()

        # Update temperature based on rate of cooling
        temperature = temperature * (1 - coolingRate)

        # Decide whether to stay or proceed with new tour
        if (newQuality < currentQuality) or (probability >= randNum):
            currentTour = newTour
            currentQuality = newQuality
            end = timer()
            trace.append([end, currentQuality])

        # Update best solution
        if currentQuality < bestQuality:
            bestTour = currentTour
            bestQuality = currentQuality

    print("finalTour: ", bestTour)
    print("finalQuality: ", bestQuality)

    return bestQuality, bestTour, trace


def LS2(nodes, time, seed):
    '''
    Genetic algorithm.

    Args:
    -    nodes: N x 3 array of nodes
    -    time: cut-off time in seconds
    -    seed: random seed

    Returns:
    -    quality: quality of best solution found
    -    tour: list of node IDs
    -    trace: list of best found solution at that point in time. Record every time a new improved solution is found.
    '''

    start = timer()
    trace = []
    np.random.seed(seed)

    # Parameters
    populationSize = 100  # How many different routes at one time, always maintain this size
    mutationRate = 0.1  # How often to mutate
    numElites = 10  # Get the top k from each generation and directly let them into the next generation
    numGenerations = 200  # Basically the number of iterations

    # Generate a certain number of initial random tours
    # Population contains INDICES of nodes
    population = []
    for i in range(populationSize):
        tour = np.random.permutation(np.arange(len(nodes)))
        population.append(tour)
    print("population: ", len(population))

    bestQuality = evaluation_function(population[i], nodes)
    bestTour = population[0]

    print("firstQuality: ", bestQuality)
    print("firstTour: ", bestTour)

    # START LOOP
    for i in range(numGenerations):
        # Stop if time has exceeded limit
        end = timer()
        if end >= time:
            print("TIME LIMIT")
            break

        # Calculate a parallel array containing quality of each tour in population
        distances = []
        fitness = []  # Inverse of distance
        for tour in population:
            quality = evaluation_function(tour, nodes)
            distances.append(quality)
            fitness.append(1 / quality)

        # Sort tours from shortest to longest
        sortInd = np.argsort(distances)  # Ascending order, shortest to longest

        newPopulation = []
        parents = []

        # Select mating pool
        # Elitism - Top k tours make it into new population automatically
        for k in range(numElites):
            newPopulation.append(population[sortInd[k]])
            parents.append(population[sortInd[k]])

        # Option 1 - Fitness Proportion Selection
        fitnessNorm = fitness / np.sum(fitness)
        samples = np.random.choice(len(population), size=populationSize - numElites, p=fitnessNorm)  # replace = ?
        for i in range(len(samples)):
            parents.append(population[samples[i]])

        # Option 2 - Tournament Selection

        # Breed - Crossover
        for i in range(populationSize - numElites):
            child = crossover(parents[i], parents[len(parents) - i - 1])
            newPopulation.append(child)

        # Mutate
        mutatedPopulation = []
        randNum = np.random.random()
        for tour in newPopulation:
            if randNum < mutationRate:
                mutatedPopulation.append(mutate(tour))
            else:
                mutatedPopulation.append(tour)

        population = mutatedPopulation

        # Find best tour of this generation
        distances = []
        for tour in population:
            quality = evaluation_function(tour, nodes)
            distances.append(quality)
        minInd = np.argmin(distances)
        minQuality = distances[minInd]
        minTour = population[minInd]

        if minQuality < bestQuality:
            bestQuality = minQuality
            bestTour = minTour
            end = timer()
            trace.append([end, bestQuality])

    # END LOOP

    print("bestQuality: ", bestQuality)
    print("bestTour: ", bestTour)

    return bestQuality, bestTour, trace


def crossover(parent1, parent2):
    '''
    Breed two tours by taking a random slice of one parent and
    joining it with the remaining genes of the second parent.
    '''
    child1 = []
    child2 = []
    gene1 = np.random.choice(np.arange(len(parent1)))
    gene2 = np.random.choice(np.arange(len(parent1)))
    startGene = min(gene1, gene2)
    endGene = max(gene1, gene2)

    for i in range(startGene, endGene):
        child1.append(parent1[i])

    child2 = [gene for gene in parent2 if gene not in child1]

    return child1 + child2


def mutate(tour):
    '''
    Randomly swap two nodes in the tour
    '''
    mutatedTour = tour.copy()
    randIndices = np.random.choice(np.arange(len(mutatedTour)), size=2)
    save = mutatedTour[randIndices[1]]
    mutatedTour[randIndices[1]] = mutatedTour[randIndices[0]]
    mutatedTour[randIndices[0]] = save
    return mutatedTour


def evaluation_function(tour, nodes):
    '''
    Calculates the total distance of a tour.

    Args:
    -    tour: N array of node INDICES

    Returns:
    -    distance: an int representing the length of the tour
    '''

    # if len(tour) == 1?

    dist = 0
    currentCity = None
    nextCity = None
    for i in range(len(tour)):
        currentCity = nodes[tour[i]]
        if i == len(tour) - 1:  # If i points to the last city in the list
            nextCity = nodes[tour[0]]
        else:
            nextCity = nodes[tour[i + 1]]
        dist += distance(currentCity[1], currentCity[2], nextCity[1], nextCity[2])
    return dist


def distance(x1, y1, x2, y2):
    '''
    Returns the distance between two cities given the x and y coordinates of both.

    Args:
    -    x1, y1: Coordinates of current city
    -    x2, y2: Coordinates of next city

    Returns:
    -    distance: Distance between the two cities, rounded to the nearest integer
    '''

    return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


if __name__ == '__main__':

    # This is sort of hardcoded right now, change it later to use argparse
    # Confused about their arguments
    # For now, assume input is of format: python tsp_main.py <FILEPATH> <ALG> <TIME> <SEED>
    # Example: tsp_main.py Atlanta.tsp BnB 120 0
    nodes = []
    args = sys.argv
    filepath = 'DATA/'
    filename = args[1]
    filepath = filepath + filename
    f = open(filepath, 'r')
    lines = f.readlines()

    for line in lines:
        splitStr = line.split()
        # print(len(splitStr))
        if len(splitStr) != 0 and splitStr[0].isdigit():
            node = [float(splitStr[0]), float(splitStr[1]), float(splitStr[2])]
            nodes.append(node)  # Can speed up by pre-allocating & indexing instead of append
    f.close()

    alg = args[2]
    time = args[3]

    seed = None
    if len(args) == 5:
        seed = int(args[4])

    # Provide invalid input checking??? e.g. alg = approx w/o seed, BnB with seed, etc
    # Can we assume valid input always?

    # Call different method based on 'alg' parameter
    if alg == "BnB":
        quality, tour, trace = BnB(nodes, time)
    elif alg == "Approx":
        quality, tour, trace = Approx(nodes, time, seed)
    elif alg == "LS1":
        quality, tour, trace = LS1(nodes, time, seed)
    elif alg == "LS2":
        quality, tour, trace = LS2(nodes, time, seed)

    # Output Files
    # Solution file
    outputName = filename[:-4] + "_" + alg + "_" + time
    if seed != None:
        outputName = outputName + "_" + str(seed)
    outputName = outputName + ".sol"

    f = open(outputName, "w")
    f.write(str(quality) + "\n")
    f.write(str(tour))
    f.close()

    # Trace file
    traceName = filename[:-4] + "_" + alg + "_" + time
    if seed != None:
        traceName = traceName + "_" + str(seed)
    traceName = traceName + ".trace"

    f = open(traceName, "w")
    for item in trace:
        f.write(str(item)[1:-1] + "\n")
    f.close()
