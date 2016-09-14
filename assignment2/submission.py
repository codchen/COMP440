from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean that your algorithm is correct.
  """

  
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
	
      gameState.isWin():
        Returns True if it's a winning state
	
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
        
      It is recommended you have separate functions: value(), max_value(), and min_value() as in the slides
      and call these functions here to make your code understandable.
    """

    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    v, move = self.value(gameState, self.depth)
    return move
    # END_YOUR_CODE

  def isTerminal(self, s):
    return s.isWin() or s.isLose() or len(s.getLegalActions()) == 0

  def value(self, s, d, agentIndex=0):
    if d == 0 or self.isTerminal(s):
      return self.evaluationFunction(s), None
    elif agentIndex == 0:
      return self.maxValue(s, d)
    else:
      return self.minValue(s, d, agentIndex)

  def maxValue(self, s, d):
    v = float('-inf')
    bestMove = None
    legalMoves = s.getLegalActions()
    nextAgent = 1 % s.getNumAgents()
    for move in legalMoves:
      if move == Directions.STOP:
        continue
      succVal = self.value(s.generatePacmanSuccessor(move),d-1 if nextAgent == 0 else d,1)[0]
      if succVal > v:
        v = succVal
        bestMove = move
    return v, bestMove

  def minValue(self, s, d, agentIndex):
    v = float('inf')
    bestMove = None
    legalMoves = s.getLegalActions(agentIndex)
    nextAgent = (agentIndex + 1) % s.getNumAgents()
    for move in legalMoves:
      succVal = self.value(s.generateSuccessor(agentIndex, move),d-1 if nextAgent == 0 else d,nextAgent)[0]
      if succVal < v:
        v = succVal
        bestMove = move
    return v, bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
      
      The same methods used in MinimaxAgent should also be useful here   
      
      It is recommended you have separate functions: value(), max_value(), and min_value() as in the slides
      and call these functions here to make the code clear   
    """

    # BEGIN_YOUR_CODE (around 45 lines of code expected)
    v, move = self.value(gameState, self.depth)
    return move
    # END_YOUR_CODE

  def isTerminal(self, s):
    return s.isWin() or s.isLose() or len(s.getLegalActions()) == 0

  def value(self, s, d, lowerBound=float('-inf'), upperBound=float('inf'), agentIndex=0):
    if d == 0 or self.isTerminal(s):
      return self.evaluationFunction(s), None
    elif agentIndex == 0:
      return self.maxValue(s, d, lowerBound, upperBound)
    else:
      return self.minValue(s, d, lowerBound, upperBound, agentIndex)

  def maxValue(self, s, d, lowerBound, upperBound):
    bestMove = None
    legalMoves = s.getLegalActions()
    nextAgent = 1 % s.getNumAgents()
    for move in legalMoves:
      if move == Directions.STOP:
        continue
      succVal = self.value(s.generatePacmanSuccessor(move),d-1 if nextAgent == 0 else d,lowerBound,upperBound,nextAgent)[0]
      if succVal > lowerBound:
        lowerBound = succVal
        bestMove = move
      if lowerBound >= upperBound:
        return upperBound, None
    return lowerBound, bestMove

  def minValue(self, s, d, lowerBound, upperBound, agentIndex):
    bestMove = None
    legalMoves = s.getLegalActions(agentIndex)
    nextAgent = (agentIndex + 1) % s.getNumAgents()
    for move in legalMoves:
      succVal = self.value(s.generateSuccessor(agentIndex, move),d-1 if nextAgent == 0 else d,lowerBound,upperBound,nextAgent)[0]
      if succVal < upperBound:
        upperBound = succVal
        bestMove = move
      if upperBound <= lowerBound:
        return lowerBound, None
    return upperBound, bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
      
      The same methods used in MinimaxAgent should also be useful here   
      
      It is recommended you have separate functions: value(), max_value(), and expect_value() as in the slides
      and call these functions here to make the code clear
    """

    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    v, move = self.value(gameState, self.depth)
    return move
    # END_YOUR_CODE

  def isTerminal(self, s):
    return s.isWin() or s.isLose() or len(s.getLegalActions()) == 0

  def value(self, s, d, agentIndex=0):
    if d == 0 or self.isTerminal(s):
      return self.evaluationFunction(s), None
    elif agentIndex == 0:
      return self.maxValue(s, d)
    else:
      return self.expValue(s, d, agentIndex)

  def maxValue(self, s, d):
    v = float('-inf')
    bestMove = None
    legalMoves = s.getLegalActions()
    nextAgent = 1 % s.getNumAgents()
    for move in legalMoves:
      if move == Directions.STOP:
        continue
      succVal = self.value(s.generatePacmanSuccessor(move),d-1 if nextAgent == 0 else d,1)[0]
      if succVal > v:
        v = succVal
        bestMove = move
    return v, bestMove

  def expValue(self, s, d, agentIndex):
    v = 0.0
    legalMoves = s.getLegalActions(agentIndex)
    nextAgent = (agentIndex + 1) % s.getNumAgents()
    for move in legalMoves:
      v += self.value(s.generateSuccessor(agentIndex, move),d-1 if nextAgent == 0 else d,nextAgent)[0]
    return v / len(legalMoves), random.choice(legalMoves)


import search
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (problem 4).

    DESCRIPTION:
      The evaluation function calculates a score which is a linear combination of the 
      features with different weights.
      Features:
        number of food left
        distance to nearest food
        distance to nearest ghost
        number of capsules left
        ghost scared time
      Since the priority goal is to let pacman win, the first 3 features have higher
      weights. The second goal is to get higher score (by eating capsules and ghosts),
      the last 2 features have smaller weights.

      Number of food left: 
        Since the pacman only wins by eating all the food, the
        number of food left has a relatively high negative weight so that the more
        food pacman eats, the better.
      Distance to nearest food:
        The closer to the food, the better. The distance is calculated
        using UCS search and use a simple weight / distance function to
        reflect the property that the closer to food, the higher the score should be.
      Distance to nearest ghost:
        The pacman dies if the it collides with the ghost. The farther the pacman
        is to the ghosts, the better. The distance to the nearest ghost is calculated
        by choosing the minimum of the manhattanDistance between the pacman and all
        the ghosts, since it is the most dangerous agent. A special check is applied
        that if the distance gets close to less than 3, there will be a large penalty
        to the score.
      Number of capsules left:
        A small penalty is applied if there are more capsules left. This is to
        encourage the pacman to eat capsules to get some time of safety.
      Ghost scared time:
        If the ghost is scared, the closer the pacman is to the ghost, the better chance
        the ghost will be eaten. So there is a relatively high positive weight associated
        with the scared time. However, if the scared time is about to expire, it is better
        for the pacman to start avoiding the ghost. So a special check for the scared time
        is applied.
  """
  if (currentGameState.isWin()):
    return 9999
  elif (currentGameState.isLose()):
    return -9999
  score = currentGameState.getScore()
  numGhost = currentGameState.getNumAgents() - 1
  problemFood = search.createPacmanFoodSearchProblem(currentGameState)
  algorithm = search.UniformCostSearch()
  algorithm.solve(problemFood)
  # Find the distance to nearest food using UCS.
  distanceToFood = algorithm.totalCost
  numCap = len(currentGameState.getCapsules())
  numFoodLeft = currentGameState.getNumFood()
  # Small penalty to number of capsules left to encourage eating it.
  score -= numCap * 15.0
  if distanceToFood != None:
    score += 10.0 / distanceToFood
  distanceToGhost = 999999
  bonus = False
  if numGhost > 0:
    for i in range(numGhost):
      ghostState = currentGameState.getGhostState(i + 1)
      distance = util.manhattanDistance(ghostState.getPosition(), currentGameState.getPacmanPosition())
      # Ghost is scared!
      if ghostState.scaredTimer > 0:
        if not bonus:
          bonus = True
          # If the ghost is scared, it is better to get close to the ghost and try to eat it.
          if (ghostState.scaredTimer > 3 and distance < ghostState.scaredTimer):
            score += 20.0 / distance * 5.0
          elif (ghostState.scaredTimer <= 3):
            # The ghost is about to get back to normal, start avoiding
            score -= 10.0 * 5.0 / ghostState.scaredTimer
          else:
            # It is not possible to eat the within scared time since it is too far.
            continue
      else:
        # Find the distance to the nearest ghost
        if (distance < distanceToGhost):
          distanceToGhost = distance
    if (distanceToGhost != 999999):
      # About to be eaten, large penalty.
      if (distanceToGhost <= 1):
        score -= 99999
      # Large penalty to very close ghosts, the closer, the larger the penalty
      elif (distanceToGhost <= 3):
        score -= 100 * 10.0 / distanceToGhost
      else:
        score -= 15.0 / distanceToGhost
  return score
  # BEGIN_YOUR_CODE (around 50 lines of code expected) 
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction


