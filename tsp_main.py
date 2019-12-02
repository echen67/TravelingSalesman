import sys
import argparse
from timeit import default_timer as timer
import numpy as np
import math
from operator import itemgetter
from random import randrange
from collections import defaultdict

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
	# Dummy values
	quality = 0
	tour = [1, 2, 3]
	trace = [[3.45, 102], [7.94, 95]]
	return quality, tour, trace


def get_neighbor(distances, tour, node):
	neighbors = distances[node]
	current_dist = [(i, j) for i, j in neighbors.items()
					if i not in tour]

	neighbor_info = sorted(current_dist, key=itemgetter(1))

	return neighbor_info[0][0], neighbor_info[0][1]


def Approx(nodes, time, seed):
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
			trace.append([round(timer(), 2), rand_quality - quality])

	quality += distances_dict[tour[0]][tour[-1]]
	trace.append([round(timer(), 2), quality])

	return quality, ','.join(map(str, tour)), trace


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
	currentTour = np.random.permutation(np.arange(len(nodes)))		# contains INDEX of nodes
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
		probability = np.exp(-(newQuality - currentQuality) / temperature)		# FIX OVERFLOW WARNING
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
	populationSize = 100					# How many different routes at one time, always maintain this size
	mutationRate = 0.1						# How often to mutate
	numElites = 10							# Get the top k from each generation and directly let them into the next generation
	numGenerations = 200					# Basically the number of iterations

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
		if i == len(tour) - 1:		# If i points to the last city in the list
			nextCity = nodes[tour[0]]
		else:
			nextCity = nodes[tour[i+1]]
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

	return round(math.sqrt((x2-x1)**2 + (y2-y1)**2))

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
			nodes.append(node)				# Can speed up by pre-allocating & indexing instead of append
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