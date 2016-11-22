"""
Text classification
"""

import util
import operator
from collections import Counter

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 3 lines of code expected) 
        self.blacklist = Counter()
        listed_num = k
        if k == -1 or k > len(blacklist):
            listed_num = len(blacklist)
        for i in range(listed_num):
            self.blacklist[blacklist[i]] += 1
        self.n = n
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        words = text.split()
        listed_num = 0
        for word in words:
            if word in self.blacklist:
                listed_num += 1
            if listed_num >= self.n:
                return -1
        return 1
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    res = Counter()
    words = x.split()
    for word in words:
        res[word] += 1
    return res  
    # END_YOUR_CODE


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        features = self.featureFunction(x)
        wsum = 0
        for feature in features:
            if feature in self.params:
                wsum += features[feature] * self.params[feature]
        return wsum
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('pos', 'neg'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    w = Counter()
    feature_list = []
    for x,_ in trainExamples:
        feature_list.append(featureExtractor(x))
    for i in range(iters):
        for j in range(len(trainExamples)):
            features = feature_list[j]
            y = trainExamples[j][1]
            wsum = 0
            for feature in features:
                if feature in w:
                    wsum += w[feature] * features[feature]
            if wsum >= 0 and y == labels[1]:
                for feature in features:
                    w[feature] -= features[feature]
            if wsum < 0 and y == labels[0]:
                for feature in features:
                    w[feature] += features[feature]
    return w
    # END_YOUR_CODE

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    res = Counter()
    features = ["."] + x.split()
    punc = [".","!","?"]
    for i in range(0, len(features)-1):
        res[features[i+1]] += 1
        if features[i] in punc:
            res["-BEGIN-" + " " + features[i+1]] += 1
        else:
            res[features[i] + " " + features[i+1]] += 1
    return res
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.classifiers = {}
        for label, classifier in classifiers:
            self.classifiers[label] = classifier      
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        scores = self.classify(x)
        label = scores[0][0]
        bestScore = scores[0][1]
        for i in range(1, len(scores)):
            if scores[i][1] > bestScore:
                label = scores[i][0]
                bestScore = scores[i][1]
        return label     
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        res = []
        for label in self.classifiers:
            res.append((label,self.classifiers[label].classify(x)))
        return res     
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    res = []
    for label in labels:
        newTrainExamples = []
        newLabels = ("pos","neg")
        for x, y in trainExamples:
            if y == label:
                newTrainExamples.append((x,"pos"))
            else:
                newTrainExamples.append((x,"neg"))
        res.append((label, WeightedClassifier(newLabels,featureFunction,learnWeightsFromPerceptron(newTrainExamples, featureFunction, newLabels, perClassifierIters))))
    return res
    # END_YOUR_CODE

