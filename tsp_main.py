import random
import sys
import argparse
from timeit import default_timer as timer
import numpy as np
import math
import random
from operator import itemgetter
from random import randrange
from collections import defaultdict

# Global variable for BnB algorithm
BnBbestQuality = float('inf')
BnBbestTour = []
BnBtrace = []
visited = []
distanceList = []

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
    global BnBbestQuality
    global BnBtrace
    global visited
    global distanceList

    start = timer()

    # Initialize the visited List and the current path list
    visited = [False] * len(nodes)
    tempPath = [-1] * len(nodes)

    # Calculate the first and second minimum distance from one node to any other node
    distanceList = [None] * len(nodes)
    for j in range(len(nodes)):
        distanceList[j] = [minimum(nodes[j], nodes), second_minimum(nodes[j], nodes)]

    # Initialize the lower bound for the algorithm
    lb = 0
    for k in range(len(nodes)):
        lb += (distanceList[k][0] + distanceList[k][1]) / 2

    # Initialize the random start point
    randNum = random.randint(0, len(nodes))
    visited[randNum - 1] = True
    tempPath[0] = randNum - 1

    # Start recursion from level 1 and weight 0
    BnBHelp(nodes, start, 1, lb, 0, tempPath, float(time))
    return BnBbestQuality, BnBbestTour, BnBtrace


def BnBHelp(nodes, time, level, lb, weight, tempPath, timeLimit):
    global BnBbestQuality
    global BnBbestTour
    global BnBtrace
    global visited
    global distanceList

    # Finalize values after all nodes are traversed
    if level == len(nodes):
        newQuality = weight + distance(nodes[tempPath[level - 1]][1], nodes[tempPath[level - 1]][2],
                                       nodes[tempPath[0]][1], nodes[tempPath[0]][2])

        if newQuality < BnBbestQuality:
            BnBbestTour = tempPath
            BnBbestQuality = newQuality
            BnBtrace.append(["%.2f" % round(timer(), 2), BnBbestQuality])

            print(newQuality)
            # print(BnBtrace)
            # print(BnBbestTour, '\n')
            return

    # If there are still nodes not traversed, check whether we should continue iterating further
    for i in range(len(nodes)):
        if timer() - time > timeLimit:
            print("Time Limit met")
            break

        if not visited[i]:
            temp_lb = lb
            weight += distance(nodes[tempPath[level - 1]][1], nodes[tempPath[level - 1]][2],
                               nodes[i][1], nodes[i][2])
            if level == 1:
                lb -= (distanceList[tempPath[level - 1]][0] + distanceList[i][0]) / 2
            else:
                lb -= (distanceList[tempPath[level - 1]][1] + distanceList[i][0]) / 2

            # Calculate the lower bound for the current node
            if lb + weight < BnBbestQuality:
                tempPath[level] = i
                visited[i] = True
                BnBHelp(nodes, time, level + 1, lb, weight, tempPath, timeLimit)

            # Change the weight and lower bound back to the level that has lb+weight < best quality
            weight -= distance(nodes[tempPath[level - 1]][1], nodes[tempPath[level - 1]][2],
                               nodes[i][1], nodes[i][2])
            lb = temp_lb
            visited = [False] * len(nodes)
            for j in range(level):
                visited[tempPath[j]] = True


# Find the minimum distance from one node to any other node
def minimum(node, nodes):
    curr_min = float('inf')
    for i in range(len(nodes)):
        curr_dist = distance(nodes[i][1], nodes[i][2], node[1], node[2])
        if curr_dist < curr_min and nodes[i] != node:
            curr_min = curr_dist
    return curr_min


# Find the second minimum distance from one node to any other node
def second_minimum(node, nodes):
    first_min, second_min = float('inf'), float('inf')
    for i in range(len(nodes)):
        if nodes[i] == node:
            continue
        curr_dist = distance(nodes[i][1], nodes[i][2], node[1], node[2])
        if curr_dist <= first_min:
            second_min = first_min
            first_min = curr_dist
        elif curr_dist <= second_min and curr_dist != first_min:
            second_min = curr_dist
    return second_min

def get_neighbor(distances, tour, node):
	neighbors = distances[node]
	current_dist = [(i, j) for i, j in neighbors.items()
					if i not in tour]

	neighbor_info = sorted(current_dist, key=itemgetter(1))

	return neighbor_info[0][0], neighbor_info[0][1]


def Approx(nodes, time=600, seed=0):
	'''
	Closest Insertion
	Args:
	-   nodes: N x 3 array of nodes
	-   time: cut-off time in seconds
	-   seed: random seed

	Returns:
	-    quality: quality of best solution found
	-    tour: list of node IDs
	-    trace: list of best found solution at that point in time. Record every time a new improved solution is found.
	'''

	# change index numbers to start from 0
	for id in nodes:
		id[0] = int(id[0]-1)

	# check if seed exist
	if seed is not None:
		# Use the seed to get a random initial tour
		np.random.seed(seed)
		random.seed(seed)
		rand_tour = np.random.permutation(np.arange(len(nodes)))

		# Evaluate distance of this random initial tour
		rand_quality = evaluation_function(rand_tour, nodes)

	# calculate distances between each node
	distances_dict = defaultdict(dict)

	for i in nodes:
		i_id = i[0]
		for j in nodes:
			j_id = j[0]
			if j_id not in distances_dict[i_id]:
				this_distance = distance(i[1], i[2], j[1], j[2])
				distances_dict[i_id][j_id] = this_distance
				distances_dict[j_id][i_id] = this_distance

	# start algorithm
	trace = []

	if seed is not None:
		trace.append([round(timer(), 2), rand_quality])

	city = randrange(1, len(nodes))     # pick random city
	tour, tours = [city], []

	neighbor, length = get_neighbor(distances_dict, tour, city)     # get nearest neighbor
	tour.append(neighbor)
	quality = length

	while len(tour) != len(nodes) and timer() <= int(time):
		select, d = None, float('inf')
		# selection - find a node k that is not in partial tour
		# closest to any node j in partial tour that minimizes d(k,j)
		for k in nodes:
			k_id = int(k[0])
			if k_id in tour:
				continue
			neighbor, length = get_neighbor(distances_dict, tour, k_id)
			if length < d:
				select, d = k_id, length
		# insertion - find the edge {i,j}
		# that minimizes d(i,k) + d(k,j) - d(i,j) and insert k
		insert_id, d = None, float('inf')
		tour = tour + [tour[0]]

		for i in range(len(tour) - 1):
			insert = distances_dict[tour[i]][select] + \
					 distances_dict[select][tour[i + 1]] - \
					 distances_dict[tour[i]][tour[i + 1]]
			if insert < d:
				insert_id, d = i, insert

		quality += distances_dict[tour[insert_id]][select] + \
				   distances_dict[select][tour[insert_id + 1]] - \
				   distances_dict[tour[insert_id]][tour[insert_id + 1]]

		tours.append(tour)
		tour.insert(insert_id + 1, select)
		tour = tour[:-1]

		if seed is not None:
			trace.append(["%.2f" % round(timer(), 2), rand_quality - quality])

	quality += distances_dict[tour[0]][tour[-1]]
	trace.append([round(timer(), 2), quality])

	# return quality, ','.join(map(str, tour)), trace
	return quality, tour, trace

def LS1(nodes, time=600, seed=0):
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

	# STILL NEED TO OPTIMIZE
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
	currentTour = np.random.permutation(np.arange(len(nodes)))		# contains INDEX of nodes
	# print("firstTour: ", currentTour)

	# Evaluate distance of this random initial tour
	currentQuality = evaluation_function(currentTour, nodes)
	print("firstQuality: ", currentQuality)

	# Keep track of best solution
	bestTour = currentTour
	bestQuality = currentQuality

	while (temperature > 0.99):
		# Stop if time has exceeded limit
		end = timer()
		if end-start >= time:
			print("TIME LIMIT")
			break

		# Create new tour using 2-opt exchange.
		randPoints = np.random.choice(np.arange(len(currentTour)), size=4, replace=False)
		newTour = currentTour.copy()
		save2 = newTour[randPoints[1]]
		newTour[randPoints[1]] = newTour[randPoints[0]]
		newTour[randPoints[0]] = save2
		# save4 = newTour[randPoints[3]]
		# newTour[randPoints[3]] = newTour[randPoints[2]]
		# newTour[randPoints[2]] = save4
		newQuality = evaluation_function(newTour, nodes)

		# Calculate probability based on temperature
		probability = np.exp(-(newQuality - currentQuality) / temperature)		# FIX OVERFLOW WARNING
		randNum = np.random.random()

		# Update temperature based on rate of cooling
		temperature = temperature * (1 - coolingRate)

		# Decide whether to stay or proceed with new tour
		if (newQuality < currentQuality) or (probability >= randNum):
			currentTour = newTour
			currentQuality = newQuality
			end = timer()
			trace.append(["%.2f" % round(end, 2), currentQuality])

		# Update best solution
		if currentQuality < bestQuality:
			bestTour = currentTour
			bestQuality = currentQuality

	# print("finalTour: ", bestTour)
	print("finalQuality: ", bestQuality)
	print("End time: ", timer())

	bestTour = list(bestTour)

	return bestQuality, bestTour, trace


def LS2(nodes, time=600, seed=0):
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
	# populationSize = 100					# How many different routes at one time, always maintain this size
	# mutationRate = 0.05						# How often to mutate
	# numElites = 30							# Get the top k from each generation and directly let them into the next generation
	# numGenerations = 500					# Basically the number of iterations

	# # CHANGE INITIAL PARAMETERS BASED ON SIZE OF INPUT?
	# if len(nodes) > 50:
	# 	print("50 NODES")
	# 	populationSize = 150
	# 	mutationRate = 0.05
	# 	numElites = 75
	# 	numGenerations = 1000
	# if len(nodes) > 75:
	# 	print("75 NODES")
	# 	populationSize = 200
	# 	mutationRate = 0.05
	# 	numElites = 75
	# 	numGenerations = 1000
	# if len(nodes) > 100:
	# 	print("100 NODES")
	# 	populationSize = 150
	# 	mutationRate = 0.01
	# 	numElites = 50
	# 	numGenerations = 1000

	# Parameters as functions of input size
	populationSize = len(nodes) * 3
	mutationRate = 0.05
	numElites = int(populationSize/3)
	numGenerations = 1000

	if len(nodes) > 50:
		print("OVER 50")
		populationSize = int(len(nodes) * 2)
		mutationRate = 0.01
		numElites = int(populationSize/2)
		numGenerations = 1000
	if len(nodes) > 100:
		print("OVER 100")
		populationSize = int(len(nodes) * 3)
		mutationRate = 0.01
		numElites = int(populationSize/2)
		numGenerations = 1000

	print(populationSize, mutationRate, numElites, numGenerations)

	# Generate a certain number of initial random tours
	# Population contains INDICES of nodes
	population = []
	for i in range(populationSize):
		tour = np.random.permutation(np.arange(len(nodes)))
		population.append(tour)

	bestQuality = evaluation_function(population[0], nodes)
	bestTour = population[0]

	print("firstQuality: ", bestQuality)
	# print("firstTour: ", bestTour)

	# Stopping criterion - no improvement for ??? seconds
	lastTime = timer()

	# START LOOP
	for i in range(numGenerations):
		# Stop if time has exceeded limit
		end = timer()
		if end-start >= time:
			print("TIME LIMIT")
			break

		# Calculate a parallel array containing quality of each tour in population
		distances = []
		fitness = []			# Inverse of distance
		for tour in population:
			quality = evaluation_function(tour, nodes)
			distances.append(quality)
			fitness.append(1/quality)

		# Sort tours from shortest to longest
		sortInd = np.argsort(distances)		# Ascending order, shortest to longest

		newPopulation = []
		parents = []

		# Select mating pool
		# Elitism - Top k tours make it into new population automatically
		for k in range(numElites):
			newPopulation.append(population[sortInd[k]])
			parents.append(population[sortInd[k]])

		# Option 1 - Fitness Proportion Selection
		fitnessNorm = fitness / np.sum(fitness)
		samples = np.random.choice(len(population), size=populationSize - numElites, p=fitnessNorm)		# replace = ?
		for i in range(len(samples)):
			parents.append(population[samples[i]])

		# Option 2 - Tournament Selection

		# Breed - Crossover
		for i in range(populationSize - numElites):
			child = crossover(parents[i], parents[len(parents)-i-1])
			newPopulation.append(child)

		# Mutate
		mutatedPopulation = []
		# randNum = np.random.random()
		for tour in newPopulation:
			randNum = np.random.random()
			if randNum < mutationRate:
				mutatedPopulation.append(mutate(tour))
			else:
				mutatedPopulation.append(tour)

		population = mutatedPopulation

		# Find best tour of this generation
		# distances = []
		# for tour in population:
		# 	quality = evaluation_function(tour, nodes)
		# 	distances.append(quality)
		# minInd = np.argmin(distances)
		# minQuality = distances[minInd]
		# minTour = population[minInd]

		# faster
		minTour = None
		minQuality = float('inf')
		for tour in population:
			d = evaluation_function(tour, nodes)
			if d < minQuality:
				minQuality = d
				minTour = tour

		# Update best solution
		if minQuality < bestQuality:
			bestQuality = minQuality
			bestTour = minTour
			end = timer()
			# trace.append([round(end, 2), bestQuality])
			trace.append(["%.2f" % round(end, 2), bestQuality])
			lastTime = timer()

		# Stopping criterion: No improvement for ?? seconds
		now = timer()
		if now-lastTime > 3:
			print("CONVERGE")
			break

	# END LOOP

	print("finalQuality: ", bestQuality)
	# print("finalTour: ", bestTour)
	print("End time: ", timer())

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

	# ARGPARSE
	parser = argparse.ArgumentParser()
	parser.add_argument('-inst')
	parser.add_argument('-alg')
	parser.add_argument('-time')
	parser.add_argument('-seed')
	parsed_args = parser.parse_args()

	filename = parsed_args.inst
	alg = parsed_args.alg
	time = parsed_args.time
	seed = parsed_args.seed

	time = int(time)
	if seed is not None:
		seed = int(seed)

	# Open file and read in nodes
	nodes = []
	filepath = 'DATA/' + filename
	f = open(filepath, 'r')
	lines = f.readlines()

	for line in lines:
		splitStr = line.split()
		if len(splitStr) != 0 and splitStr[0].isdigit():
			node = [float(splitStr[0]), float(splitStr[1]), float(splitStr[2])]
			nodes.append(node)				# Can speed up by pre-allocating & indexing instead of append
	f.close()

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
	outputName = filename[:-4] + "_" + alg + "_" + str(time)
	if seed != None and alg != "BnB":
		outputName = outputName + "_" + str(seed)
	outputName = outputName + ".sol"

	f = open(outputName, "w")
	f.write(str(quality) + "\n")
	temp = str(tour)
	temp = temp[1:-1]
	temp = temp.replace(" ", "")
	f.write(temp)
	f.close()

	# Trace file
	traceName = filename[:-4] + "_" + alg + "_" + str(time)
	if seed != None and alg != "BnB":
		traceName = traceName + "_" + str(seed)
	traceName = traceName + ".trace"

	f = open(traceName, "w")
	for item in trace:
		temp = str(item)[1:-1] + "\n"
		temp = temp.replace(" ", "")
		temp = temp.replace("'", "")
		f.write(temp)
	f.close()