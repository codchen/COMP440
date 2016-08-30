import graderUtil, util

grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Problem 3-1a: ucsTestCase

def testUCSTestCase(n):
    # Just make sure the test case is valid
    ucs = util.UniformCostSearch()
    ucs.solve(submission.createUCSTestCase(n))
    if ucs.actions and len(ucs.actions) <= 2 and ucs.numStatesExplored >= n:
        grader.assignFullCredit()
    else:
        grader.fail("Your test case did not meet the specifications")

grader.addBasicPart('3-1a', lambda : testUCSTestCase(3),5)


############################################################
# Problem 3-1b: astarReduction

def testZeroHeuristic():
    # Make sure putting the zero heuristic in doesn't change the problem.
    problem1 = util.trivialProblem
    problem2 = submission.astarReduction(problem1, lambda state : 0)
    grader.requireIsEqual(problem1.startState(), problem2.startState())

    for state in ['A', 'B', 'C']:
        if not grader.requireIsEqual(problem1.isGoal(state), problem2.isGoal(state)): return
        if not grader.requireIsEqual(problem1.succAndCost(state), problem2.succAndCost(state)): return
grader.addBasicPart('3-1b-0', testZeroHeuristic,10)



############################################################
# Problem 3-2b

def testDelivery():
    problem = submission.DeliveryProblem(util.deliveryScenario0)
    algorithm = util.UniformCostSearch()
    algorithm.solve(problem)
    grader.requireIsEqual(5, algorithm.totalCost)
grader.addBasicPart('3-2b', testDelivery,5)

def testNonTrivialHeuristic():
    # Make sure putting a non-trivial heuristic in does not change the solution.
    problem = util.trivialProblem
    heuristic = util.trivialProblemHeuristic
    ucs = util.UniformCostSearch()
    astar = submission.AStarSearch(heuristic)
    ucs.solve(problem)
    astar.solve(problem)
    if ucs.actions == astar.actions and ucs.totalCost == astar.totalCost:
        grader.assignFullCredit()
    else:
        grader.fail("Reduced problem has different solution!")
grader.addBasicPart('3-1b-1', testNonTrivialHeuristic,10)



############################################################
# Problem 3-2c

def testHeuristic1():
    scenario = util.deliveryScenario1
    problem = submission.DeliveryProblem(scenario)
    algorithm = submission.AStarSearch(submission.createHeuristic1(scenario))
    algorithm.solve(problem)
    if algorithm.totalCost != 26:
        grader.fail("heuristic1 produces wrong total cost")
        return
    # This is a coarse check, report your exact number of explored nodes in writeup
    if algorithm.numStatesExplored >= 61:
        grader.fail("heuristic1 explores too many states")
        return
    grader.assignFullCredit()
grader.addBasicPart('3-2c-0', testHeuristic1,10)


############################################################
# Problem 3-2d

def testHeuristic2():
    scenario = util.deliveryScenario2
    problem = submission.DeliveryProblem(scenario)
    algorithm = submission.AStarSearch(submission.createHeuristic2(scenario, 0))
    algorithm.solve(problem)
    if algorithm.totalCost != 25:
        grader.fail("heuristic2 produces wrong total cost")
        return
    # This is a coarse check, report your exact number of explored nodes in writeup
    if algorithm.numStatesExplored >= 40:
        grader.fail("heuristic2 explores too many states")
        return
    grader.assignFullCredit()
grader.addBasicPart('3-2d-0', testHeuristic2,10)


############################################################
# Problem 3-2e

def testHeuristic3():
    scenario = util.deliveryScenario3
    problem = submission.DeliveryProblem(scenario)
    algorithm = submission.AStarSearch(submission.createHeuristic3(scenario))
    algorithm.solve(problem)
    if algorithm.totalCost != 27:
        grader.fail("heuristic3 produces wrong total cost")
        return
    # This is a coarse check, report your exact number of explored nodes in writeup
    if algorithm.numStatesExplored >= 60:
        grader.fail("heuristic3 explores too many states")
        return
    grader.assignFullCredit()
grader.addBasicPart('3-2e-0', testHeuristic3, 10)


grader.grade()
