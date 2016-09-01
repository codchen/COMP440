import re, util, collections

############################################################
# Problem 1a: UCS test case

# Return an instance of util.SearchProblem.
# You might find it convenient to use
# util.createSearchProblemFromString.
def createUCSTestCase(n):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    problemStr = ""
    for i in range(n - 1):
        problemStr += "0 " + str(i + 2) + " 1\n"
    for i in range(n - 1):
        problemStr += str(i + 2) + " 1 2\n"
    problemStr += "0 1 2"
    return util.createSearchProblemFromString("0", "1", problemStr)
    # END_YOUR_CODE

############################################################
# Problem 1b: A-star search

# Takes the SearchProblem |problem| you're trying to solve and a |heuristic|
# (which is a function that maps a state to an estimate of the cost to the
# goal).  Returns another search problem |newProblem| such that running uniform
# cost search on |newProblem| is equivalent to running A* on |problem| with
# |heuristic|.
def astarReduction(problem, heuristic):
    class NewSearchProblem(util.SearchProblem):
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
class AStarSearch(util.SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # Reduce the |problem| to |newProblem|, which is solved by UCS.
        newProblem = astarReduction(problem, self.heuristic)
        algorithm = util.UniformCostSearch()
        algorithm.solve(newProblem)

        # Copy solution back
        self.actions = algorithm.actions
        if algorithm.totalCost != None:
            self.totalCost = algorithm.totalCost + self.heuristic(problem.startState())
        else:
            self.totalCost = None
        self.numStatesExplored = algorithm.numStatesExplored

############################################################
# Problem 2b: Delivery

class DeliveryProblem(util.SearchProblem):
    # |scenario|: delivery specification.
    def __init__(self, scenario):
        self.scenario = scenario
    # Return the start state.
    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        # list of packages carrying, list of packages left
        return (self.scenario.truckLocation, (-1, ) * self.scenario.numPackages)
        # END_YOUR_CODE

    # Return whether |state| is a goal state or not.
    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return state[0] == self.scenario.truckLocation and state[1] == (1, ) * self.scenario.numPackages
        # END_YOUR_CODE

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state):
        # Hint: Call self.scenario.getNeighbors((x,y)) to get the valid neighbors
        # at that location. In order for the simulation code to work, please use
        # the exact strings 'Pickup' and 'Dropoff' for those two actions.
        # BEGIN_YOUR_CODE (around 18 lines of code expected)
        neighbors = self.scenario.getNeighbors(state[0])
        result = []
        for action, loc in neighbors:
            c = collections.Counter(state[1])
            result.append((action, (loc, state[1]), 1 + c[0]))
            if (state[0] in self.scenario.pickupLocations):
                index = self.scenario.pickupLocations.index(state[0])
                if (state[1][index] == -1):
                    newPackageState = list(state[1])
                    newPackageState[index] = 0
                    result.append(('Pickup', (state[0], tuple(newPackageState)), 0))
            if (state[0] in self.scenario.dropoffLocations):
                index = self.scenario.dropoffLocations.index(state[0])
                if (state[1][index] == 0):
                    newPackageState = list(state[1])
                    newPackageState[index] = 1
                    result.append(('Dropoff', (state[0], tuple(newPackageState)), 0))
        return result
        # END_YOUR_CODE

############################################################
# Problem 2c: heuristic 1


# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers and not do any deliveries,
# you just need to go home
def createHeuristic1(scenario):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return abs(scenario.truckLocation[0] - state[0][0]) + abs(scenario.truckLocation[1] - state[0][1])
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2d: heuristic 2

# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers, but
# you'll need to deliver the given |package|, and then go home
def createHeuristic2(scenario, package):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        pickup = scenario.pickupLocations[package]
        dropoff = scenario.dropoffLocations[package]
        if (state[1][package] == -1):
            dist_pickup = abs(state[0][0] - pickup[0]) + abs(state[0][1] - pickup[1])
            dist_dropoff = 2 * (abs(dropoff[0] - pickup[0]) + abs(dropoff[1] - pickup[1]))
            dist_home = abs(dropoff[0] - scenario.truckLocation[0]) + abs(dropoff[1] - scenario.truckLocation[1])
            return dist_pickup + dist_dropoff + dist_home
        elif (state[1][package] == 0):
            dist_dropoff = 2 * (abs(dropoff[0] - state[0][0]) + abs(dropoff[1] - state[0][1]))
            dist_home = abs(dropoff[0] - scenario.truckLocation[0]) + abs(dropoff[1] - scenario.truckLocation[1])
            return dist_dropoff + dist_home
        else:
            return abs(state[0][0] - scenario.truckLocation[0]) + abs(state[0][1] - scenario.truckLocation[1])
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2e: heuristic 3

# Return a heuristic corresponding to solving a relaxed problem
# where you will delivery the worst(i.e. most costly) |package|,
# you can ignore all barriers.
# Hint: you might find it useful to call
# createHeuristic2.
def createHeuristic3(scenario):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    def heuristic(state):
        return max([createHeuristic2(scenario, i)(state) for i in range(scenario.numPackages)])
    return heuristic
    # END_YOUR_CODE