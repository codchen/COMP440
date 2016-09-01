import random
directions = [(1,0),(-1,0),(0,1),(0,-1)]
def randomStart(size):
	trace = [(0,0)]
	dirChoice = [list(directions)]
	resDirs = []
	while len(trace) < size:
		curIdx = len(trace) - 1
		if len(dirChoice[curIdx]) == 0:
			del trace[curIdx]
			del dirChoice[curIdx]
			del resDirs[curIdx - 1]
			continue
		attemptDir = random.choice(dirChoice[curIdx])
		del dirChoice[curIdx][dirChoice[curIdx].index(attemptDir)]
		newNode = (trace[curIdx][0] + attemptDir[0], trace[curIdx][1] + attemptDir[1])
		if newNode not in trace:
			trace.append(newNode)
			dirChoice.append(list(directions))
			resDirs.append(attemptDir)
	return trace, tuple(resDirs)

import operator
def evalAndSort(states, H):
	dirs2Cost = {}
	for dir, trace in states.items():
		dirs2Cost[dir] = cost(trace, H)
	return sorted(dirs2Cost.items(), key=operator.itemgetter(1))

def isValid(trace):
	for node in trace:
		if trace.count(node) > 1:
			return False
	return True

import math
def cost(trace, H):
	if not isValid(trace):
		return 1000000
	res = 0
	for i in range(len(trace)):
		for j in range(i + 1, len(trace)):
			if H[i] == 1 and H[j] == 1:
				res += math.hypot(trace[i][0]-trace[j][0], trace[i][1]-trace[j][1])
	return res

def crossover(dirs1, dirs2):
	trace = [(0,0)]
	resDirs = ()
	for i in range(int(0.2 * len(dirs1))):
		resDirs += (dirs1[i],)
		newNode = (trace[i][0] + dirs1[i][0], trace[i][1] + dirs1[i][1])
		trace.append(newNode)
	for i in range(int(0.2 * len(dirs1)), int(0.8 * len(dirs1))):
		resDirs += (dirs2[i],)
		newNode = (trace[i][0] + dirs2[i][0], trace[i][1] + dirs2[i][1])
		trace.append(newNode)
	for i in range(int(0.8 * len(dirs1)), int(len(dirs1))):
		resDirs += (dirs1[i],)
		newNode = (trace[i][0] + dirs1[i][0], trace[i][1] + dirs1[i][1])
		trace.append(newNode)
	return trace, resDirs

def dirs2Trace(dirs):
	trace = [(0,0)]
	for x,y in dirs:
		newNode = (trace[len(trace)-1][0]+x,trace[len(trace)-1][1]+y)
		trace.append(newNode)
	return trace

# 0: (1,0), 1: (-1,0), 2: (0,1), 3: (0,-1)
def mutate(dirs):
	candidates = range(4 * len(dirs))
	for i in range(len(dirs)):
		del candidates[candidates.index(i + len(dirs) * directions.index(dirs[i]))]
	idx = random.choice(candidates)
	tmpDirs = list(dirs)
	tmpDirs[idx%len(dirs)] = directions[idx/len(dirs)]
	return dirs2Trace(tmpDirs), tuple(tmpDirs)

def iter(gen, sequence, r):
	mutationHelper = range(10)
	sortedDirs = evalAndSort(gen, sequence)
	tmpGen = {}
	firstHalf = range(len(sortedDirs) / 2 + 1)
	for i in firstHalf:
		tmpGen[sortedDirs[i][0]] = gen[sortedDirs[i][0]]
	nonElite = set([])
	while len(tmpGen) < len(gen):
		child = crossover(sortedDirs[random.choice(firstHalf)][0],sortedDirs[random.choice(range(len(sortedDirs)))][0])
		while not isValid(child[0]) or child[1] in tmpGen:
			child = crossover(sortedDirs[random.choice(firstHalf)][0],sortedDirs[random.choice(range(len(sortedDirs)))][0])
		tmpGen[child[1]] = child[0]
		nonElite.add(child[1])
	for child1 in nonElite:
		if random.choice(mutationHelper) < 4:
			before = len(tmpGen)
			mutated = mutate(child1)
			while not isValid(mutated[0]) or mutated[1] in tmpGen:
				mutated = mutate(child1)
			tmpGen[mutated[1]] = mutated[0]
			if (len(tmpGen) > before):
				del tmpGen[child1]
	print(sortedDirs[0][1], dirs2Trace(sortedDirs[0][0]))
	return tmpGen

def run(genNum, popSize, sequence, r=500):
	gen = {}
	for i in range(popSize):
		trace, dir = randomStart(len(sequence))
		gen[dir] = trace
	for i in range(genNum):
		gen = iter(gen, sequence, r)

run(200, 200, [0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0]) 
