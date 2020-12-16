import numpy as np
import cv2
import math

def check_boundary(y, x, bound_y, bound_x, delta):
    return (0 <= y - delta and y + delta <= bound_y) \
           and (0 <= x - delta and x + delta <= bound_x)

def normalize_vectors(fv):
    fv_norm = np.linalg.norm(fv, axis=1).reshape(-1, 1)
    fv_normalize = fv / fv_norm
    return fv_normalize

def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    # Calculate image gradients
    img_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    img_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    # magnitude & direction of gradients in image
    img_mag = np.sqrt(img_y * img_y + img_x * img_x)
    img_dir = np.arctan2(img_y, img_x) * 180 / np.pi
    img_dir = np.where(img_dir >= 0.0, img_dir, img_dir + 360)
    img_dir_hist = np.zeros(shape=img_dir.shape, dtype=np.int)
    # print(np.max(img_dir), np.min(img_dir))

    # Divide into 8 bins in img_dir. Mapping as follows.
    # 0 <= angle < 45 maps to 1, ..., 315 <= angle < 360 maps to 8
    for i in range(8):
        low_theta = 45.0 * i
        high_theta = 45.0 * (i + 1)
        img_dir_hist = np.where(np.logical_and(img_dir >= low_theta, img_dir < high_theta),
                                i + 1, img_dir_hist)
    assert np.sum(img_dir_hist == 0) == 0, 'img_dir_hist should have no 0s'

    fv = []
    delta = feature_width // 2
    m, n = image.shape
    for i in range(len(x)):
        cur_y = int(round(y[i])) + 1
        cur_x = int(round(x[i])) + 1
        if not check_boundary(cur_y, cur_x, m, n, delta):
            print('This should not happen!!!')
            exit()
        fov = img_dir_hist[cur_y - delta:cur_y + delta, cur_x - delta:cur_x + delta]
        fov_mag = img_mag[cur_y - delta:cur_y + delta, cur_x - delta:cur_x + delta]
        fv_sample = np.array([], dtype=np.float)
        for b in range(4):
            for a in range(4):
                start_y = 4 * b
                start_x = 4 * a
                hist = np.zeros((8,), dtype=np.float)
                small_fov = fov[start_y:start_y + 4, start_x:start_x + 4]
                small_fov_mag = fov_mag[start_y:start_y + 4, start_x:start_x + 4]
                # Composite histogram
                for j in range(8):
                    # hist[j] = np.sum(small_fov == j + 1).astype(np.float)
                    hist[j] = np.sum(np.where(small_fov == j + 1, small_fov_mag, 0.0))
                # assert np.sum(hist) == 16, 'the sum of histogram should be 16'
                fv_sample = np.concatenate((fv_sample, hist))
        assert len(fv_sample) == 128, 'the length of fv_sample should be 128.'
        fv.append(fv_sample)

    fv = np.array(fv, dtype=np.float)
    # normalization
    fv = normalize_vectors(fv)
    # solve illumination problem
    fv = np.where(fv >= 0.2, 0.2, fv).astype(np.float)
    # normalization
    fv = normalize_vectors(fv)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


''' The case of simple patch 16 x 16 as feature vector
    m, n = image.shape
    num_interest = len(x)
    fv = []
    delta = feature_width // 2

    for i in range(num_interest):
        # print(x.shape, x[i], type(x[i]))
        int_x = int(round(x[i])) + 1
        int_y = int(round(y[i])) + 1
        if (0 <= int_x - delta and int_x + delta <= n) and (0 <= int_y - delta and int_y + delta <= m):
            patch = image[int_y - delta:int_y + delta, int_x - delta:int_x + delta].flatten()
            fv.append(patch)
    fv = np.array(fv, dtype=np.float)
'''