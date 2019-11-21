import sys
import argparse
from timeit import default_timer as timer
import numpy as np
import math

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

	start = timer()
	trace = []
	coolingRate = 0.0001
	temperature = 10000

	# Use the seed to get a random initial tour
	print("nodes: ", len(nodes))
	np.random.seed(seed)
	currentTour = np.random.permutation(np.arange(len(nodes)))		# contains INDEX of nodes
	print("currentTour: ", currentTour)

	# Evaluate distance of this random initial tour
	currentQuality = evaluation_function(currentTour, nodes)
	print("currentQuality: ", currentQuality)

	# Consider a different stopping condition - iteration not based on temp, bounded by time
	while (temperature > 0.99):
		# Stop if time has exceeded limit
		end = timer()

		# Create new tour. Do you just randomly get four points? Is that considered adjacent?
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

	print("finalTour: ", currentTour)
	print("finalQuality: ", currentQuality)

	return currentQuality, currentTour, trace

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

	# Dummy values
	quality = 0
	tour = [1, 2, 3]
	trace = [[3.45, 102], [7.94, 95]]
	return quality, tour, trace

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
	-    distance: Distance between the two cities
	'''

	return math.sqrt((x2-x1)**2 + (y2-y1)**2)

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