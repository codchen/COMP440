#!/usr/bin/env python
"""
run as python grader.py 
"""


import math, random, copy

import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')

from collections import Counter

import util
import nerUtils
from util import Counters
import itertools as it


############################################################
# Problem 3.1: CRFs
############################################################

simpleCRF = submission.LinearChainCRF( ["-FEAT-", "-SIZE-"], 
        submission.binaryFeatureFunction,
        Counter({
            ("-FEAT-", "-SIZE-") : 0.8,
            ("-SIZE-", "-FEAT-") : 0.5,
            ("-SIZE-", "-SIZE-") : 1.,
            ("-FEAT-", "Beautiful") : 1.,
            ("-SIZE-", "Beautiful") : 0.5,
            ("-FEAT-", "2") : 0.5,
            ("-SIZE-", "2") : 1.0,
            ("-FEAT-", "bedroom") : 0.5,
            ("-SIZE-", "bedroom") : 1.0,}) )
exampleInput = "Beautiful 2 bedroom".split()
exampleTags = "-FEAT- -SIZE- -SIZE-".split()

import pickle
states, parameters = pickle.load( open('data/english.binary.crf') ) 
englishCRF = submission.LinearChainCRF( states, submission.binaryFeatureFunction, parameters ) 


############################################################
# -- 3.1a: Viterbi decoding

def part31a_0():
    """Check that you get the gold labelling."""
    xs = exampleInput
    ys = exampleTags
    ys_ = submission.computeViterbi(simpleCRF, xs)
    grader.requireIsEqual( ys, ys_ )
grader.addBasicPart('3.1a-0', part31a_0, 10)



############################################################
# -- 3.1b: Forward backward

def part31b_0():
    """
    Check that the forward algorithm's output is normalized at each iteration
    """
    xs = exampleInput
    _, forward = submission.computeForward(simpleCRF, xs)
    for i in xrange(len(xs)):
        grader.requireIsEqual( 1.0, sum( forward[i].values() ) )
grader.addBasicPart('3.1b-0', part31b_0, 2)

def part31b_1():
    """
    Check that the backward algorithm's output is normalized at each iteration
    """
    xs = exampleInput
    backward = submission.computeBackward(simpleCRF, xs)
    for i in xrange(len(xs)):
        grader.requireIsEqual( 1.0, sum( backward[i].values() ) )
grader.addBasicPart('3.1b-1', part31b_1, 0)

def part31b_2():
    """Check that values match our reference"""
    xs = exampleInput
    z = 5.881
    forward = [
            Counter({'-FEAT-': 0.622, '-SIZE-': 0.377}), 
            Counter({'-SIZE-': 0.761, '-FEAT-': 0.238}), 
            Counter({'-SIZE-': 0.741, '-FEAT-': 0.258})]
    
    z_, forward_ = submission.computeForward(simpleCRF, xs)
    for vec, vec_ in zip( forward, forward_):
        grader.requireIsTrue( Counters.approximateEquals( vec, vec_ ) )
    grader.requireIsEqual( z, z_, 1e-2)
grader.addBasicPart('3.1b-2', part31b_2, 8)

def part31b_3():
    """Check that values match our reference"""
    xs = exampleInput
    backward = [
            Counter({'-SIZE-': 0.564, '-FEAT-': 0.435}),
            Counter({'-SIZE-': 0.567, '-FEAT-': 0.432}),
            Counter({'-FEAT-': 0.5, '-SIZE-': 0.5})]
    backward_ = submission.computeBackward(simpleCRF, xs)
    for vec, vec_ in zip( backward, backward_):
        grader.requireIsTrue( Counters.approximateEquals( vec, vec_ ) )
grader.addBasicPart('3.1b-3', part31b_3, 0)


############################################################
# -- 3.1c: Compute expected statistics

def part31c_0():
    """Check that the expected counts at each step add up to 1."""
    xs = exampleInput
    T = submission.computeEdgeMarginals(simpleCRF, xs)
    for t in T:
        grader.requireIsEqual( 1.0, sum(t.values()) )
grader.addBasicPart('3.1c-0', part31c_0, 1)

def part31c_1():
    """Check that values match our reference"""
    xs = exampleInput
    T = [ Counter({('-BEGIN-', '-FEAT-'): 0.561, ('-BEGIN-', '-SIZE-'): 0.439}),
          Counter({('-FEAT-', '-SIZE-'): 0.463, ('-SIZE-', '-SIZE-'): 0.343, 
                   ('-SIZE-', '-FEAT-'): 0.096, ('-FEAT-', '-FEAT-'): 0.096}),
          Counter({('-SIZE-', '-SIZE-'): 0.590, ('-SIZE-', '-FEAT-'): 0.217,
                   ('-FEAT-', '-SIZE-'): 0.151, ('-FEAT-', '-FEAT-'): 0.041})
        ]
    T_ = submission.computeEdgeMarginals(simpleCRF, xs)
    for t, t_ in zip(T, T_):
        grader.requireIsTrue( Counters.approximateEquals(t, t_) )
grader.addBasicPart('3.1c-1', part31c_1, 9)



############################################################
# -- 3.2a: NER Features

def part32a_0():
    """Check that you have all the features we expect"""
    xs = exampleInput
    phi = Counter({('-BEGIN-', '-FEAT-'): 1.0, ('-FEAT-', 'Beautiful'): 1.0, ('-FEAT-', 'PREV:-BEGIN-'): 1.0, ('-FEAT-', 'NEXT:2'): 1.0, ('-FEAT-', '-CAPITALIZED-'): 1.0, ('-FEAT-', '-POST-CAPITALIZED-'): 0.0})
    phi_ = submission.nerFeatureFunction(0, '-BEGIN-', '-FEAT-', xs)
    grader.requireIsTrue( Counters.approximateEquals(phi, phi_) )

    phi = Counter({('-FEAT-', '-SIZE-'): 1.0, ('-SIZE-', 'PREV:Beautiful'): 1.0, ('-SIZE-', 'NEXT:bedroom'): 1.0, ('-SIZE-', '-PRE-CAPITALIZED-'): 1.0, ('-SIZE-', '2'): 1.0, ('-SIZE-', '-POST-CAPITALIZED-'): 0.0, ('-SIZE-', '-CAPITALIZED-'): 0.0})
    phi_ = submission.nerFeatureFunction(1, '-FEAT-', '-SIZE-', xs)
    grader.requireIsTrue( Counters.approximateEquals(phi, phi_) )
    
    phi = Counter({('-SIZE-', '-SIZE-'): 1.0, ('-SIZE-', 'PREV:2'): 1.0, ('-SIZE-', 'bedroom'): 1.0, ('-SIZE-', 'NEXT:-END-'): 1.0, ('-SIZE-', '-CAPITALIZED-'): 0.0, ('-SIZE-', '-PRE-CAPITALIZED-'): 0.0})
    phi_ = submission.nerFeatureFunction(2, '-SIZE-', '-SIZE-', xs)
    grader.requireIsTrue( Counters.approximateEquals(phi, phi_) )

grader.addBasicPart('3.2a-0', part32a_0, 10)

############################################################
# -- 3.3: Gibbs sampling for CRF

def getCRFBlocks(xs):
    """
    Groups variables into blocks that are updated simultaneously.
    In this case, each variable belongs in its own block.
    @params xs - observation sequence
    """
    return range(len(xs))

def part33a_0():
    """
    The sample distribution and exact distribution should match fairly closely.
    """
    xs = exampleInput
    N = 10000

    difference = 0.0
    for ys, estimatedProb in submission.computeGibbsProbabilities( simpleCRF, 
            getCRFBlocks,
            submission.chooseGibbsCRF,
            xs, N ).iteritems():
        trueProb = nerUtils.computeProbability( simpleCRF, xs, ys )
        difference = abs( trueProb - estimatedProb )
        grader.requireIsLessThan( 5e-2, difference )
grader.addBasicPart('3.3a', part33a_0, 5, 5)

def part33a_1():
    """
    The estimated best sequence should be the right one!
    """
    xs = exampleInput
    ys = exampleTags
    N = 10000

    ys_ = submission.computeGibbsBestSequence(
            simpleCRF,
            getCRFBlocks,
            submission.chooseGibbsCRF,
            xs, 
            N)
    grader.requireIsEqual( ys, ys_ )

grader.addBasicPart('3.3b', part33a_1, 5, 5)

def part33a_3():
    """
    Your implementation should be fast enough. Make sure you are only using
    factors in the Markov blanket.
    * Done correctly, this takes about 0.5 seconds. 
    * Done naively, it can take ~2.5 seconds.
    """
    xs = "Werner & Co entered court today . Werner maintained that they were not guilty .".split()
    N = 10000

    submission.computeGibbsProbabilities( englishCRF,
            getCRFBlocks,
            submission.chooseGibbsCRF,
            xs, N )
    grader.requireIsTrue(True)

grader.addBasicPart('3.3c', part33a_3, 5, 5)





if __name__ == "__main__":
    grader.grade()
