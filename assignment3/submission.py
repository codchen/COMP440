import collections, util, math, random, copy

############################################################
# Problem 3.1.1

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    Q = 0
    for (newState, prob, reward) in mdp.succAndProbReward(state, action):
        Q += (reward + mdp.discount() * V[newState]) * prob
    return Q
    # END_YOUR_CODE

############################################################
# Problem 3.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 8 lines of code expected)
    old_V = V.copy()
    improve = True
    while improve:
        improve = False
        new_V = copy.copy(old_V)
        for state in mdp.states:
            new_V[state] = computeQ(mdp, old_V, state, pi[state])
            if not abs(old_V[state] - new_V[state]) < epsilon:
                improve = True
        old_V = new_V
    return old_V
    # END_YOUR_CODE

############################################################
# Problem 3.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    policy = {}
    values = {}
    for state in mdp.states:
        for action in mdp.actions(state):
            values[action] = computeQ(mdp, V, state, action)
        policy[state] = max(values, key=lambda k: values[k])
    return policy
    # END_YOUR_CODE

############################################################
# Problem 3.1.4

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute |V| and |pi|, which should both be dicts
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        changed = True
        V = dict.fromkeys(mdp.states, 0)
        pi = {}
        policy = computeOptimalPolicy(mdp, V)
        while changed:
            changed = False
            V = policyEvaluation(mdp, V, policy, epsilon)
            new_policy= computeOptimalPolicy(mdp, V)
            if policy != new_policy:
                policy = new_policy
                changed = True
        # END_YOUR_CODE
        for state in policy:
            if isinstance(state, tuple) and len(state) == 3 and state[2] == (0,):
                continue
            else:
                pi[state] = policy[state]
        self.pi = pi
        self.V = V

############################################################
# Problem 3.1.5

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 11lines of code expected)
        improve = True
        pi = {}
        V = dict.fromkeys(mdp.states, 0)
        while improve:
            improve = False
            policy = computeOptimalPolicy(mdp, V)
            new_V = dict.fromkeys(mdp.states, 0)
            for state in mdp.states:
                new_V[state] = computeQ(mdp, V, state, policy[state])
                if not abs(V[state] - new_V[state]) < epsilon:
                    improve = True
            V = copy.copy(new_V)
        # END_YOUR_CODE
        # This is to get rid of the terminal states in the policy for blackjack.
        for state in policy:
            if isinstance(state, tuple) and len(state) == 3 and state[2] == (0,):
                continue
            else:
                pi[state] = policy[state]
        self.pi = pi
        self.V = V

############################################################
# Problem 3.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        self.state = 1
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 1
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return ['left', 'right']
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        if (state == self.startState()):
            if (action == 'left'):
                return [(0, 0.3, 10), (2, 0.7, 1)]
            else:
                return [(2, 1.0, 1)]
        else:
            return []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 1.0
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return 0.1
    # END_YOUR_CODE

############################################################
# Problem 3.2.1

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.  The second element is the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.  The final element
    # is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to (0,).
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 40 lines of code expected)
        if (state[0] > self.threshold or state[2] == (0,)):
            return []
        if (action == 'Quit'):
            newState = (state[0], None, (0,))
            return [((state[0], None, (0,)), 1.0, state[0])]
        if (action == 'Peek' and state[1] != None):
            return []
        totalCards = sum(state[2])
        states = []
        if (action == 'Peek'):
            for i in range(len(self.cardValues)):
                if (state[2][i] > 0):
                    newState = (state[0], i, state[2])
                    prob = float(state[2][i]) / totalCards
                    reward = -self.peekCost
                    states.append((newState, prob, reward))
            return states
        if (action == 'Take'):
            cards = list(state[2])
            # Peek on the previous round
            if (state[1] != None):
                cards[state[1]] -= 1
                value = state[0] + self.cardValues[state[1]]
                if (value > self.threshold):
                    newState = (value, None, (0,))
                    return [(newState, 1.0, 0)]
                else:
                    newState = (value, None, tuple(cards))
                    # not any(cards) is True iff cards are all 0
                    if not any(cards):
                        newState = (value, None, (0,))
                        return [(newState, 1.0, newState[0])]
                    else:
                        return [(newState, 1.0, 0)]
            for i in range(len(self.cardValues)):
                if (state[2][i] > 0):
                    cards = list(state[2])
                    cards[i] -= 1
                    value = state[0] + self.cardValues[i]

                    if (value > self.threshold):
                        newState = (value, None, (0,))
                        states.append((newState, float(state[2][i]) / totalCards, 0))
                    else:
                        newState = (value, None, tuple(cards))
                        if not any(cards):
                            states.append((newState, float(state[2][i]) / totalCards, newState[0]))
                        else:
                            states.append((newState, float(state[2][i]) / totalCards, 0))
            return states
        else:
            raise Exception("Invalid action.")
        # END_YOUR_CODE

    def discount(self):
        return 1.0

############################################################
# Problem 3.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    return  BlackjackMDP([9, 10, 11, 19], 1, 20, 1)
    # END_YOUR_CODE
