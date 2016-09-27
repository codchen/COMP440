import util, submission, copy

class RobotMDP(util.MDP):
	def startState(self):
		return 2

	def actions(self, state):
		if (state == 1 or state == 3):
			return ['c', 'r', 'e']
		else:
			return ['c', 'r']

	def succAndProbReward(self, state, action):
		result = []
		if (state == 1):
			if (action == 'c'):
				result.append((1, 0.5, 10))
				result.append((2, 0.25, 4))
				result.append((3, 0.25, 8))
			elif (action == 'r'):
				result.append((1, 0.0625, 8))
				result.append((2, 0.75, 2))
				result.append((3, 0.1875, 4))
			elif (action == 'e'):
				result.append((1, 0.25, 4))
				result.append((2, 0.125, 6))
				result.append((3, 0.625, 4))
			else:
				raise Exception("ERROR in state 1")
		elif (state == 2):
			if (action == 'c'):
				result.append((1, 0.5, 14))
				result.append((3, 0.5, 18))
			elif (action == 'r'):
				result.append((1, 0.0625, 8))
				result.append((2, 0.875, 16))
				result.append((3, 0.0625, 8))
			else:
				raise Exception("ERROR in state 2")
		else:
			if (action == 'c'):
				result.append((1, 0.25, 10))
				result.append((2, 0.25, 2))
				result.append((3, 0.5, 8))
			elif (action == 'r'):
				result.append((1, 0.125, 6))
				result.append((2, 0.75, 4))
				result.append((3, 0.125, 2))
			elif (action == 'e'):
				result.append((1, 0.75, 4))
				result.append((2, 0.0625, 0))
				result.append((3, 0.1875, 8))
			else:
				raise Exception("ERROR in state 3")
		return result

	def discount(self):
		return 0.1


class ValueIteration(util.MDPAlgorithm):
	def solve(self, mdp):
		mdp.computeStates()
		i = 0
		V = dict.fromkeys(mdp.states, 0)
		values = {}
		pi = {}
		while (True):
			i += 1
			new_V = copy.copy(V)
			new_pi = {}
			for state in mdp.states:
				for action in mdp.actions(state):
					values[action] = submission.computeQ(mdp, V, state, action)
				new_pi[state] = max(values, key=lambda k: values[k])
				new_V[state] = values[new_pi[state]]
			if (new_pi == pi):
				V = new_V
				break
			else:
				print "Iteration: " + str(i)
				V = new_V
				pi = new_pi
				print "New V: "
				print V
				print "New policy: "
				print pi
		print "V:"
		print V
		print "Optimal Policy: "
		print pi

mdp = RobotMDP()
vi = ValueIteration()
vi.solve(mdp)
