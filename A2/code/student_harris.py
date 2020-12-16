import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def plot_img_gradient(img, img_x, img_y, step):
    '''
    Plot the image gradient. The step decides the image set to plot in Harris process.
    If step = 0, (img, grad_x(img), grad_y(img))
    else if step = 1, A = (grad_x(img)^2, grad_y(img)^2, grad_x(img) * grad_y(img))
    else if step = 2, Gaussian(A)
    '''
    plot_set = [['image', 'I_x', 'I_y'], ['I_x^2', 'I_y^2', 'I_xI_y'],
                ['g(I_x^2)', 'g(I_y^2)', 'g(I_xI_y)']]
    title_set = plot_set[step]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 3, 1)
    # ax.axis("off")
    ax.set_title(title_set[0])
    plt.imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
    ax = plt.subplot(1, 3, 2)
    # ax.axis("off")
    ax.set_title(title_set[1])
    plt.imshow(img_x, cmap='gray', vmin=np.min(img_x), vmax=np.max(img_x))
    ax = plt.subplot(1, 3, 3)
    # ax.axis("off")
    ax.set_title(title_set[2])
    plt.imshow(img_y, cmap='gray', vmin=np.min(img_y), vmax=np.max(img_y))
    plt.show()


def suppress_boundary_gradients(img, feature_width):
    rad = feature_width // 2
    img[0:rad, :] = 0.0
    img[-rad:, :] = 0.0
    img[:, 0:rad] = 0.0
    img[:, -rad:] = 0.0
    return img


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    # Calculate image gradients
    img_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    img_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    # print(image.shape, image.dtype, img_x.shape, img_y.shape)
    # plot_img_gradient(image, img_x, img_y, step=0)

    # Calculate multiplications
    img_y2 = img_y * img_y
    img_x2 = img_x * img_x
    img_xy = img_x * img_y
    # print(img_y2.shape, img_x2.shape, img_xy.shape)
    # plot_img_gradient(img_x2, img_y2, img_xy, step=1)

    # Calculate Gaussian filtered image
    sigma = 1.0
    ksize = int(2 * np.ceil(3 * sigma) + 1)  # 7 when sigma=1.0
    g_kernel_1d = cv2.getGaussianKernel(ksize=3, sigma=sigma)
    g_kernel_2d = np.outer(g_kernel_1d, g_kernel_1d.transpose())
    g_img_y2 = cv2.filter2D(img_y2, ddepth=-1, kernel=g_kernel_2d)
    g_img_x2 = cv2.filter2D(img_x2, ddepth=-1, kernel=g_kernel_2d)
    g_img_xy = cv2.filter2D(img_xy, ddepth=-1, kernel=g_kernel_2d)
    # print(g_img_x2.shape, g_img_y2.shape, g_img_xy.shape)
    # plot_img_gradient(g_img_x2, g_img_y2, g_img_xy, step=2)

    # Calculate R
    k = 0.04
    threshold = 0.01  # 0.01 best
    det_H = g_img_x2 * g_img_y2 - g_img_xy * g_img_xy
    trace_H = g_img_x2 + g_img_y2
    R = det_H - k * (trace_H * trace_H)
    # Suppress the gradients on boundary
    R = suppress_boundary_gradients(R, feature_width)
    # Threshold R and extract interest points
    R_th_bool = R > threshold
    R_th = np.where(R_th_bool, R, 0.0)
    R_th_val = R[R_th_bool]
    interest_point = np.argwhere(R_th_bool)
    y_interest = interest_point[:, 0]
    x_interest = interest_point[:, 1]
    # print(x_interest.shape, y_interest.shape, R_th_val.shape)

    # Choose the non-maximum suppression algorithm option.
    # There is one option which is 'ANMS'.
    # If option = 'ANMS' stands for adaptive non-maximum suppression.
    # Else, the default action is done which is window based NMS.
    option = 'ANMS'

    if option == 'NMS':
        coord = []
        confidences = []
        # For each interest points, if the point is local maximum of certain window,
        # keep it. Else, reject that point. Use 5x5 window here.
        window = 2
        for i in range(len(R_th_val)):
            coord_x = x_interest[i]
            coord_y = y_interest[i]
            R_val = R_th_val[i]
            local_max = np.max(R_th[coord_y - window:coord_y + window + 1,
                               coord_x - window:coord_x + window + 1])
            if R_val == local_max:
                coord.append([coord_y, coord_x])
                confidences.append(R_val)

        coord = np.array(coord, dtype=np.int)
        confidences = np.array(confidences, dtype=np.float)
        y = coord[:, 0]
        x = coord[:, 1]
        # print(x.shape, y.shape, confidences.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE (for extra credit)  #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################
    if option == 'ANMS':
        n = 5000
        print('ANMS applied.')
        # Sort the confidences
        R_th_val_sort = np.argsort(R_th_val)[::-1]
        R_th_val = R_th_val[R_th_val_sort]
        x_interest = x_interest[R_th_val_sort]
        y_interest = y_interest[R_th_val_sort]
        coord = np.concatenate((y_interest.reshape(-1, 1), x_interest.reshape(-1, 1)), axis=1)
        suppression_radii = []
        for i in range(len(x_interest)):
            cur_pos = coord[i].reshape(1, -1)
            # print(cur_pos.shape)
            if i == 0:  # global maximum case
                suppression_radii.append(1000.0)
                continue
            prev_pos = coord[0:i]
            # print(prev_pos.shape)
            dist = cdist(cur_pos, prev_pos)
            # print(dist.shape)
            suppression_radii.append(np.min(dist))
        suppression_radii = np.array(suppression_radii, dtype=np.float)
        suppression_radii_sort = np.argsort(suppression_radii)[::-1]
        suppression_radii = suppression_radii[suppression_radii_sort]
        assert suppression_radii[0] == 1000.0  # the max sup radii should be 1000.0
        x_interest = x_interest[suppression_radii_sort]
        y_interest = y_interest[suppression_radii_sort]
        R_th_val = R_th_val[suppression_radii_sort]
        x = x_interest[0:n]
        y = y_interest[0:n]
        confidences = R_th_val[0:n]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, confidences, scales, orientations

'''
Result 
        NMS      ANMS
notre   93       95
mount   96       96
gaudi            4

Before running gaudi NMS, save the result of current images.
It contains the info of gaudi ANMS images which takes very long time.
'''