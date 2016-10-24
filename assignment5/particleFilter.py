'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
import util, collections

import random
import math

# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):
    
    NUM_PARTICLES = 200
    
    # Function: Init
    # --------------
    # Constructer that initializes an ParticleFilter object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in a dict of Counters
        # self.transProbDict[oldTile][newTile] = probability of transitioning from oldTile to newTile
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.Counter()
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]
            
        # Initialize the particles randomly
        self.particles = collections.Counter()
        potentialParticles = self.transProbDict.keys()
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1
            
        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates the particle filters beliefs about the probability that the car is in each tile.
    # Particle probabilities should already sum to 1, but this function normalizes to be safe.
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    # Function: Observe:
    # - Reweight the particles based on observation
    # - Resamples
    # -----------------
    # Updates beliefs based on the distance observation $d_t$ and your position $a_t$.
    # observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    # agentX: x location of your car (not the one you are tracking)
    # agentY: y location of your car (not the one you are tracking)
    #
    # Suggestion: Loop over each particle and then update the weight in self.particles
    # with the observation's probablity.
    #
    # Note that your belief cloud, self.beliefs will be updated for you during resample(),
    # which is called at the end of this function.
    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_CODE (around 5 lines of code expected)
        for tile in self.particles:
            dist = math.sqrt((agentX - util.colToX(tile[1])) ** 2 + (agentY - util.rowToY(tile[0])) ** 2)
            self.particles[tile] *= util.pdf(dist,Const.SONAR_STD,observedDist)
        # END_YOUR_CODE
        self.resample()

    # Function: Resample
    # ---------------------
    # Resample your particles. Then update self.beliefs with the newly sampled particles.
    # util.weightedRandomChoice(weighted_dictionary) should be useful here
    #
    # Suggestion: Use a loop to create |self.NUM_PARTICLES| new particles.
    # Inside the loop, you will randomly sample a particle from self.particles
    # (use util.weightedRandomChoice).
    #
    # Note: While there are many possible implementations, for grading
    # purposes, we expect the following:
    # - the particles are looped over in the suggested order
    # - exactly 1 random number is generated per particle.
    def resample(self):
        newParticles = collections.Counter()
        # BEGIN_YOUR_CODE (around 3 lines of code expected)
        for i in range(self.NUM_PARTICLES):
            particleIndex = util.weightedRandomChoice(self.particles)
            newParticles[particleIndex] += 1
        # END_YOUR_CODE
        self.particles = newParticles
        self.updateBelief()

    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Update your inference to handle the passing of one time step.
    # Use the transition probabilities in self.transProb.
    # util.weightedRandomChoice(weighted_dictionary) should be useful here
    #
    # Suggestion: Loop over each type of particle.  For
    # |self.particles[particle]| times, pick a random transition to use for the
    # particle and add it to the newParticles counter.
    #
    # Note: While there are many possible implementations, for grading
    # purposes, we expect the following:
    # - the particles are looped over in the suggested order
    # - exactly 1 random number is generated per particle.
    def elapseTime(self):
        newParticles = collections.Counter()
        # BEGIN_YOUR_CODE (around 7 lines of code expected)
        for oldTile in self.particles:
            for i in range(self.particles[oldTile]):
                newTile = util.weightedRandomChoice(self.transProbDict[oldTile])
                newParticles[newTile] += 1
        # END_YOUR_CODE
        self.particles = newParticles
      
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief
