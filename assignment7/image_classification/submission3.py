import collections
import numpy as np

############################################################
# Problem 3.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    centroids = np.random.randn(patches.shape[0],k)
    numPatches = patches.shape[1]
    # dimension is patches.shape[0]
    for i in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        # Assignment Step
        z = np.zeros((numPatches,), dtype=np.int)

        for n in range(numPatches):
          dist = np.zeros((k, ))
          for j in range(k):
            dist[j] = np.linalg.norm(patches[:, n] - centroids[:, j])
          z[n] = np.argmin(dist)
        # Update Step
        for j in range(k):
          indices = np.where(z == j)[0]
          centroids[:, j] = np.mean(patches[:,indices], axis=1)
        # END_YOUR_CODE
    return centroids

############################################################
# Problem 3.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    for i in range(numPatches):
      feature = np.zeros((k,))
      p = patches[:, i]
      for j in range(k):
        dist = np.mean()
        for K in range(k):
          miu = centroids[:, K]
          dist += np.linalg.norm(p - miu)
        dist /= k
        dist -= np.linalg.norm(p - centroids[:, j])
        feature[j] = max(dist, 0)
      features[i] = feature
    # END_YOUR_CODE
    return features

############################################################
# Problem 3.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    d_theta = np.zeros(theta.shape)
    yy = 2 * y - 1
    for i in range(len(theta)):
      d_theta[i] =  -featureVector[i] * yy * math.exp(-theta.dot(featureVector) * yy) / (1 + math.exp(-theta.dot(featureVector) * yy))
    return d_theta
    # END_YOUR_CODE

############################################################
# Problem 3.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    d_theta = np.zeros(theta.shape)
    if (theta.dot(featureVector) * (2 * y - 1) >= 1):
     return d_theta
    for i in range(len(theta)):
      d_theta[i] = -featureVector[i] * (2 * y - 1)
    return d_theta
    # END_YOUR_CODE

