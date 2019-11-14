import sys
import argparse

def BnB(nodes, time):
	'''
	Args:
	-    nodes: N x 2 array of 
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
	-    nodes: N x 2 array of nodes
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
	Args:
	-    nodes: N x 2 array of nodes
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

def LS2(nodes, time, seed):
	'''
	Args:
	-    nodes: N x 2 array of nodes
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

if __name__ == '__main__':

	# This is sort of hardcoded right now, change it later to use argparse
	# Confused about their arguments
	# For now, assume input is of format: tsp_main.py <FILEPATH> <ALG> <TIME> <SEED>
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
		if splitStr[0].isdigit():
			node = [float(splitStr[0]), float(splitStr[1]), float(splitStr[2])]
			nodes.append(node)				# Can speed up by pre-allocating & indexing instead of append
	f.close()

	alg = args[2]
	time = args[3]

	seed = None
	if len(args) == 5:
		seed = args[4]

	# Provide invalid input checking??? e.g. alg = approx w/o seed, etc

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
		outputName = outputName + "_" + seed
	outputName = outputName + ".sol"

	f = open(outputName, "w")
	f.write(str(quality) + "\n")
	f.write(str(tour))
	f.close()

	# Trace file
	traceName = filename[:-4] + "_" + alg + "_" + time
	if seed != None:
		traceName = traceName + "_" + seed
	traceName = traceName + ".trace"

	f = open(traceName, "w")
	for item in trace:
		f.write(str(item)[1:-1] + "\n")
	f.close()