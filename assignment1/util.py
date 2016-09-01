import heapq, collections, re, sys, time, os, random

############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return whether |state| is a goal state or not.
    def isGoal(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    # First, call solve on the desired SearchProblem |problem|.
    # Then it should set two things:
    # - self.actions: list of actions that takes one from the start state to a goal
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem): raise NotImplementedError("Override me")

############################################################
# Uniform cost search algorithm (Dijkstra's algorithm).

class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose=1):
        self.verbose = verbose

    def solve(self, problem):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.startState()
        frontier.update(startState, 0)

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, pastCost = frontier.removeMin()
            if state == None: break
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print "Exploring %s with pastCost %s" % (state, pastCost)

            # Check if we've reached the goal; if so, extract solution
            if problem.isGoal(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print "numStatesExplored = %d" % self.numStatesExplored
                    print "totalCost = %s" % self.totalCost
                    print "actions = %s" % self.actions
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.succAndCost(state):
                if self.verbose >= 3:
                    print "  Action %s => %s with cost %s + %s" % (action, newState, pastCost, cost)
                if frontier.update(newState, pastCost + cost):
                    # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)
        if self.verbose >= 1:
            print "No path found"

# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left...

############################################################
# Simple examples of search problems to test your code for Problem 1.

# A simple search problem on the number line:
# Start at 0, want to go to 10, costs 1 to move down, 2 to move up.
class NumberLineSearchProblem:
    def startState(self): return 0
    def isGoal(self, state): return state == 10
    def succAndCost(self, state): return [('West', state-1, 1), ('East', state+1, 2)]

# A simple search problem on a square grid:
# Start at init position, want to go to (0, 0)
# cost 2 to move up/left, 1 to move down/right
class GridSearchProblem(SearchProblem):
    def __init__(self, size, x, y): self.size, self.start = size, (x,y)
    def startState(self): return self.start
    def isGoal(self, state): return state == (0, 0)
    def succAndCost(self, state):
        x, y = state
        results = []
        if x-1 >= 0: results.append(('North', (x-1, y), 2))
        if x+1 < self.size: results.append(('South', (x+1, y), 1))
        if y-1 >= 0: results.append(('West', (x, y-1), 2))
        if y+1 < self.size: results.append(('East', (x, y+1), 1))
        return results

# Function to create search problems from a graph.
# You can use this to test your algorithm.
#
# here is an example:
# if you want to model the following graph as a search problem:
#   A->B->C, where A->B costs 1, B->C costs 2;
#   you start off at A, and your goal is to go to C.
# 
# then for the parameters to pass in to create a search problem:
#   start is "A", goal is "C", and description is "A B 1\nB C 2"
def createSearchProblemFromString(start, goal, description):
    # Parse the graph
    graph = collections.defaultdict(list)
    for line in description.split("\n"):
        if len(line) == 0 or line.startswith('#'): continue
        # Edge from state a to state b.
        a, b, cost = line.split(" ")
        cost = float(cost)
        # Action is the same as the destination state (b).
        graph[a].append((b, b, cost))

    # Construct the search problem
    class ExplicitSearchProblem():
        def startState(self): return start
        def isGoal(self, state): return state == goal
        def succAndCost(self, state): return graph[state]
    return ExplicitSearchProblem()

# Trivial graph.
trivialProblem = createSearchProblemFromString("A", "C",
"""
A B 4
A C 8
B C 3
""")


# An admissible heuristic function for the trivial graph.
def trivialProblemHeuristic(state):
    if state == "A":
        return 7
    elif state == "B":
        return 3
    else:
        return 0

############################################################
# Supporting code for Problem 3.

# A |DeliveryScenario| contains the following information:
# - Starting and ending location of the truck.
# - Pickup and dropoff locations for each package.
# Note: all locations are represented as (row, col) pairs.
# Scenarios are used to construct DeliveryProblem (your job).
# Member variables:
# - numRows, numCols: dimensions of the grid
# - numPackages: number of packages to deliver
# - pickupLocations, dropoffLocations: array of locations for each of the packages
# - truckLocation: where you must start and end
class DeliveryScenario:
    # |description|: string representation of the delivery specification.
    def __init__(self, description):
        #print description
        self.pickupLocations = []
        self.dropoffLocations = []
        self.truckLocation = None

        def addLoc(locations, i, loc):
            while i >= len(locations): locations.append(None)
            locations[i] = loc

        self.grid = []
        for line in description.split("\n"):
            if len(line) == 0: continue
            r = len(self.grid)
            row = []
            for c, cell in enumerate(re.split('\s+', line)):
                if cell == '#':  # Barrier
                    row.append('#')
                else:  # No barrier
                    row.append('.')
                    loc = (r, c)
                    if cell.startswith('P'):
                        addLoc(self.pickupLocations, int(cell[1:]), loc)
                    elif cell.startswith('D'):
                        addLoc(self.dropoffLocations, int(cell[1:]), loc)
                    elif cell == 'T':
                        self.truckLocation = loc
                    elif cell == '.':
                        pass
                    else:
                        raise Exception("Invalid cell: %s" % cell)
            self.grid.append(row)
            if len(row) != len(self.grid[0]):
                raise Exception("Rows have different number of columns: %s and %s" % (row, self.grid[0]))
        if len(self.pickupLocations) != len(self.dropoffLocations):
            raise Exception("Number of pickup and dropoff locations don't match")
        if None in self.pickupLocations:
            raise Exception("Missing pickup location: %s" % self.pickupLocations)
        if None in self.dropoffLocations:
            raise Exception("Missing dropoff location: %s" % self.dropoffLocations)
        if self.truckLocation == None:
            raise Exception("Missing truck location: %s" % self.truckLocation)

        self.numRows = len(self.grid)
        self.numCols = len(self.grid[0])
        self.numPackages = len(self.pickupLocations)

    # Return a list of valid (action, newLoc) pairs that we can take from loc.
    def getNeighbors(self, loc):
        r, c = loc
        newActionLocations = []
        if r-1 >= 0 and self.grid[r-1][c] != '#': newActionLocations.append(('North', (r-1, c)))
        if c-1 >= 0 and self.grid[r][c-1] != '#': newActionLocations.append(('West', (r, c-1)))
        if r+1 < self.numRows and self.grid[r+1][c] != '#': newActionLocations.append(('South', (r+1, c)))
        if c+1 < self.numCols and self.grid[r][c+1] != '#': newActionLocations.append(('East', (r, c+1)))
        return newActionLocations

    def __str__(self): return self.render(self.truckLocation, ['ready'] * self.numPackages)

    def render(self, truckLoc, statuses):
        grid = [[cell for cell in row] for row in self.grid]  # Copy
        for package, loc in enumerate(self.pickupLocations):
            if statuses[package] == 'ready': grid[loc[0]][loc[1]] = 'P'+str(package)
        for package, loc in enumerate(self.dropoffLocations):
            if statuses[package] != 'done': grid[loc[0]][loc[1]] = 'D'+str(package)
        grid[truckLoc[0]][truckLoc[1]] = 'T/' + str(statuses.count('transit'))
        return "\n".join([' '.join("%-3s" % x for x in row) for row in grid])

    def simulate(self, actions, animate):
        if not actions: return
        r, c = self.truckLocation
        statuses = ['ready'] * self.numPackages
        for action in actions:
            print self.render((r, c), statuses)
            if animate: time.sleep(0.2)
            else: sys.stdin.readline()
            # Move truck
            if action == 'North':
                r -= 1
            elif action == 'West':
                c -= 1
            elif action == 'South':
                r += 1
            elif action == 'East':
                c += 1
            elif action == 'Pickup':
                for package, loc in enumerate(self.pickupLocations):
                    if loc == (r, c) and statuses[package] == 'ready':
                        statuses[package] = 'transit'
            elif action == 'Dropoff':
                for package, loc in enumerate(self.dropoffLocations):
                    if loc == (r, c) and statuses[package] == 'transit':
                        statuses[package] = 'done'
            else:
                raise Exception("Invalid action: %s" % action)
            if animate: os.system('clear')
        print self.render((r, c), statuses)

# Legend:
# - Pi: pickup location of i-th package
# - Di: dropoff location of i-th package
# - T: truck location
deliveryScenario0 = DeliveryScenario('T P0 D0')
deliveryScenario1 = DeliveryScenario("""
.  .  .  .  .  #  T
.  P0 #  #  .  #  .
D0 .  .  #  .  .  .
.  .  .  #  .  .  .
""")
deliveryScenario2 = DeliveryScenario("""
.  .  .  .  .  #  T
.  .  #  #  .  #  .
.  .  P0 #  .  .  .
.  .  .  .  .  #  D0
""")
deliveryScenario3 = DeliveryScenario("""
D0 .  .  .  .  T
.  #  #  P1 .  .
.  .  #  #  .  .
P0 .  .  .  .  D1
""")
deliveryScenario4 = DeliveryScenario("""
.  .  #  T  #  .  .
.  .  #  P0 #  .  .
.  .  #  .  #  .  .
D0 .  .  P1 .  .  D1
""")

def createRandomScenario(numRows, numCols, numPackages):
    grid = [['.'] * numCols for _ in range(numRows)]
    for _ in range(numRows * numCols / 4):
        grid[random.randint(0, numRows-1)][random.randint(0, numCols-1)] = '#'
    def setRandomFreeLocation(x):
        r = random.randint(0, numRows-1)
        c = random.randint(0, numCols-1)
        if grid[r][c] == '.': grid[r][c] = x
        else: setRandomFreeLocation(x)
    setRandomFreeLocation('T')
    for i in range(numPackages):
        setRandomFreeLocation('P' + str(i))
        setRandomFreeLocation('D' + str(i))
    return DeliveryScenario('\n'.join(' '.join(row) for row in grid))
