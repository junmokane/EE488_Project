import numpy as np
from scipy.spatial.distance import cdist

def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################
    matches = []
    confidences = []

    #print(features1.shape, features2.shape)
    m, n = features1.shape
    #print(m, n)
    dist = cdist(features1, features2)
    #print(dist.shape)
    # print(dist[0, 0:5], dist[-1, -5:])
    dist_sort = np.argsort(dist, axis=1)
    #print(dist_sort[0, 0:5], dist_sort[-1, 0:5])
    #print(dist_sort.shape)
    d1_index = dist_sort[:, 0]  # minimum distance index
    d2_index = dist_sort[:, 1]  # 2nd lowest distance index

    for i in range(m):
        d1 = dist[i, d1_index[i]]
        d2 = dist[i, d2_index[i]]
        ratio = d1 / d2
        if ratio < 0.8:
            matches.append([i, d1_index[i]])
            confidences.append(ratio)
    #print(np.min(confidences), np.max(confidences))
    # print(matches[0:3], confidences[0:3])
    # print(matches[-3:], confidences[-3:])


    matches = np.array(matches, dtype=np.int)
    # print(matches[0:5], matches[-5:])
    confidences = np.array(confidences, dtype=np.float)
    # print(matches[0:10])
    # print(confidences[0:10])

    # Sort the confidences so that matches have the ascending ratio
    confidences_sort = np.argsort(confidences)
    matches = matches[confidences_sort]
    confidences = confidences[confidences_sort]
    # print(matches[0:5])
    # print(x1[matches[:5, 0]], y1[matches[:5, 0]])
    # print(x2[matches[:5, 1]], y2[matches[:5, 1]])
    # print(confidences[0:10])

    # print(confidences.shape, matches.shape)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
