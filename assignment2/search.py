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

            # Check if we've reached the goal; if so, extract solution
            if problem.isGoal(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.succAndCost(state):
                if frontier.update(newState, pastCost + cost):
                    # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)

def astarReduction(problem, heuristic):
    class NewSearchProblem(SearchProblem):
        # Please refer to util.SearchProblem to see the functions you need to
        # overried.
        # BEGIN_YOUR_CODE (around 9 lines of code expected)
        def startState(self):
            return problem.startState()
        def isGoal(self, state):
            return problem.isGoal(state)
        def succAndCost(self, state):
            succ = problem.succAndCost(state)
            result = []
            for (action, newState, cost) in succ:
                result.append((action, newState, cost + heuristic(newState) - heuristic(state)))
            return result
        # END_YOUR_CODE
    newProblem = NewSearchProblem()
    return newProblem

# Implements A-star search by doing a reduction.
class AStarSearch(SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # Reduce the |problem| to |newProblem|, which is solved by UCS.
        newProblem = astarReduction(problem, self.heuristic)
        algorithm = UniformCostSearch()
        algorithm.solve(newProblem)

        # Copy solution back
        self.actions = algorithm.actions
        if algorithm.totalCost != None:
            self.totalCost = algorithm.totalCost + self.heuristic(problem.startState())
        else:
            self.totalCost = None
        self.numStatesExplored = algorithm.numStatesExplored


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

# A simple search problem on a square grid:
# Start at init position, want to go to (0, 0)
# cost 2 to move up/left, 1 to move down/right
class PacmanFoodSearchProblem(SearchProblem):
    def __init__(self, pos, grid, walls):
        self.start = pos
        self.grid = grid
        self.wall = walls

    def getNeighbors(self, state):
        r, c = state
        newActionLocations = []
        if r-1 >= 0 and not self.wall[r-1][c]: newActionLocations.append(('North', (r-1, c)))
        if c-1 >= 0 and not self.wall[r][c-1]: newActionLocations.append(('West', (r, c-1)))
        if r+1 < self.grid.width and not self.wall[r+1][c]: newActionLocations.append(('South', (r+1, c)))
        if c+1 < self.grid.height and not self.wall[r][c+1]: newActionLocations.append(('East', (r, c+1)))
        return newActionLocations

    def startState(self): 
        return self.start

    def isGoal(self, state):
        x, y = state
        return self.grid[x][y]

    def succAndCost(self, state):
        neighbors = self.getNeighbors(state)
        results = []
        for action, loc in neighbors:
            results.append((action, loc, 1))
        return results

class PacmanCapsulesSearchProblem(SearchProblem):
    def __init__(self, pos, capPos, walls):
        self.start = pos
        self.capPos = capPos
        self.wall = walls

    def getNeighbors(self, state):
        r, c = state
        newActionLocations = []
        if r-1 >= 0 and not self.wall[r-1][c]: newActionLocations.append(('North', (r-1, c)))
        if c-1 >= 0 and not self.wall[r][c-1]: newActionLocations.append(('West', (r, c-1)))
        if r+1 < self.wall.width and not self.wall[r+1][c]: newActionLocations.append(('South', (r+1, c)))
        if c+1 < self.wall.height and not self.wall[r][c+1]: newActionLocations.append(('East', (r, c+1)))
        return newActionLocations

    def startState(self): 
        return self.start

    def isGoal(self, state):
        x, y = state
        for x1, y1 in self.capPos:
            if x == x1 and y == y1:
                return True
        return False

    def succAndCost(self, state):
        neighbors = self.getNeighbors(state)
        results = []
        for action, loc in neighbors:
            results.append((action, loc, 1))
        return results

def createPacmanFoodSearchProblem(state):
    return PacmanFoodSearchProblem(state.getPacmanPosition(), state.getFood(), state.getWalls())

def createPacmanCapsulesSearchProblem(state):
    return PacmanCapsulesSearchProblem(state.getPacmanPosition(), state.getCapsules(), state.getWalls())

