#cython: boundscheck=False, nonecheck=False, initializedcheck=False
# Utility routines in cython for prediction in hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

import numpy as np
cimport numpy as np
np.import_array() 
from hdbscan.dist_metrics cimport DistanceMetric

from libc.float cimport DBL_MAX
from libc.math cimport exp

cpdef get_tree_row_with_child(np.ndarray tree, np.intp_t child):

    cdef np.intp_t i
    cdef np.ndarray[np.intp_t, ndim = 1] child_array = tree['child']

    for i in range(tree.shape[0]):
        if child_array[i] == child:
            return tree[i]

    return tree[0]

cpdef double min_dist_to_exemplar(np.ndarray[np.float64_t, ndim=1] point,
                                  np.ndarray[np.float64_t, ndim=2] cluster_exemplars,
                                  DistanceMetric dist_metric):
    """
    Compute minimum distance from point to any exemplar using DistanceMetric.
    """
    cdef:
        Py_ssize_t i, num_features, n_exemplars
        double min_distance = DBL_MAX
        double current_distance

        double[::1] point_view
        double[:, ::1] exemplars_view

    # Validate dimensions
    if point.ndim != 1 or cluster_exemplars.ndim != 2:
        raise ValueError("Expected point to be 1D and exemplars to be 2D.")
    if point.shape[0] != cluster_exemplars.shape[1]:
        raise ValueError("Dimensionality mismatch between point and exemplars.")

    # Views
    point_view = point
    exemplars_view = cluster_exemplars

    n_exemplars = cluster_exemplars.shape[0]
    num_features = point.shape[0]

    for i in range(n_exemplars):
        current_distance = dist_metric.dist(&point_view[0], &exemplars_view[i, 0], num_features)
        if current_distance < min_distance:
            min_distance = current_distance

    return min_distance

cdef np.ndarray[np.float64_t, ndim=1] dist_vector(
                    np.ndarray[np.float64_t, ndim=1] point,
                    list exemplars_list,
                    DistanceMetric dist_metric):

    cdef np.intp_t i
    cdef np.ndarray[np.float64_t, ndim=2] exemplars
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(len(exemplars_list))


    for i in range(len(exemplars_list)):
        exemplars = exemplars_list[i]
        point = np.ascontiguousarray(point, dtype=np.float64)
        exemplars = np.ascontiguousarray(exemplars, dtype=np.float64)
        result[i] = min_dist_to_exemplar(point, exemplars, dist_metric)

    return result

cpdef np.ndarray[np.float64_t, ndim=1] dist_membership_vector(
                    np.ndarray[np.float64_t, ndim=1] point,
                    list exemplars_list,
                    DistanceMetric dist_metric,
                    softmax=False):

    cdef np.intp_t i
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(len(exemplars_list))
    cdef np.ndarray[np.float64_t, ndim=1] vector
    cdef np.float64_t sum = 0.0

    vector = dist_vector(point, exemplars_list, dist_metric)

    if softmax:
        for i in range(vector.shape[0]):
            result[i] = 1.0 / vector[i]
        result = np.exp(result - np.nanmax(result))
        sum = np.sum(result)

    else:
        for i in range(vector.shape[0]):
            if vector[i] != 0:
                result[i] = 1.0 / vector[i]
            else:
                result[i] = DBL_MAX / vector.shape[0]
            sum += result[i]

    for i in range(result.shape[0]):
        result[i] = result[i] / sum

    return result

cpdef np.ndarray[np.float64_t, ndim=2] all_points_dist_membership_vector(
        np.ndarray[np.float64_t, ndim=2] all_points,
        list exemplars_list,
        DistanceMetric dist_metric,
        softmax=False):

    cdef np.ndarray[np.float64_t, ndim=2] result
    cdef np.intp_t i

    result = np.empty((all_points.shape[0], len(exemplars_list)),
                      dtype=np.float64)

    for i in range(all_points.shape[0]):
        result[i] = dist_membership_vector(all_points[i],
                                           exemplars_list,
                                           dist_metric,
                                           softmax)

    return result

cdef np.ndarray[np.float64_t, ndim=1] merge_height(
        np.intp_t point_cluster,
        np.float64_t point_lambda,
        np.ndarray[np.intp_t, ndim=1] clusters,
        np.ndarray cluster_tree):

    cdef np.intp_t i
    cdef np.intp_t j

    cdef np.intp_t left_cluster
    cdef np.intp_t right_cluster
    cdef int took_right_parent
    cdef int took_left_parent
    cdef np.intp_t cluster

    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(clusters.shape[0],
                                                            dtype=np.float64)
    cdef np.ndarray[np.intp_t, ndim=1] parents
    cdef np.ndarray[np.intp_t, ndim=1] children
    cdef np.ndarray[np.float64_t, ndim=1] lambdas

    # convert the cluster tree for fast direct access
    parents = cluster_tree['parent'].astype(np.intp)
    children = cluster_tree['child'].astype(np.intp)
    lambdas = cluster_tree['lambda_val'].astype(np.float64)


    for i in range(clusters.shape[0]):

        took_right_parent = False
        took_left_parent = False

        right_cluster = clusters[i]
        left_cluster = point_cluster

        while left_cluster != right_cluster:
            if left_cluster > right_cluster:
                took_left_parent = True
                last_cluster = left_cluster

                # Set left_cluster to be its parent
                for j in range(children.shape[0]):
                    if children[j] == left_cluster:
                        left_cluster = parents[j]
                        break
            else:
                took_right_parent = True
                last_cluster = right_cluster

                # Set right_cluster to be its parent
                for j in range(children.shape[0]):
                    if children[j] == right_cluster:
                        right_cluster = parents[j]
                        break

        if took_left_parent and took_right_parent:
            # Take the lambda value of last_cluster merging in
            for j in range(children.shape[0]):
                if children[j] == last_cluster:
                    result[i] = lambdas[j]
                    break
        else:
            result[i] = point_lambda

    return result


cpdef np.float64_t safe_always_positive_division(
        np.float64_t numerator,
        np.float64_t denominator):
    """ This is a helper function to divide numbers safely without getting a ZeroDivision error, the
    function handles zero division by assuming the denominator is always positive 
    
    Parameters
    ----------
    numerator: floating
        any floating point type 
    denominator: floating
        any floating point type
    
    Returns
    -------
    floating
    """
    if denominator <= 0:
        # prevent zero division or negative result
        denominator = 1e-8
    return numerator / denominator


cpdef np.ndarray[np.float64_t, ndim=1] per_cluster_scores(
        np.intp_t neighbor,
        np.float32_t lambda_,
        np.ndarray[np.intp_t, ndim=1] clusters,
        np.ndarray tree,
        dict max_lambda_dict,
        np.ndarray cluster_tree):

    cdef np.intp_t point_cluster
    cdef np.float64_t point_lambda
    cdef np.float64_t max_lambda

    cdef np.intp_t i

    cdef np.ndarray[np.float64_t, ndim=1] result

    point_row = get_tree_row_with_child(tree, neighbor)
    point_cluster = point_row['parent']
    point_lambda = lambda_
    max_lambda = max_lambda_dict[point_cluster]

    # Save an allocation by assigning and reusing result ...
    # height = merge_height(point_cluster, point_lambda,
    #                       clusters, cluster_tree)
    result = merge_height(point_cluster, point_lambda,
                          clusters, cluster_tree)

    # Cythonize: result = np.exp(-(max_lambda / height))
    for i in range(result.shape[0]):
        # result[i] = exp(-(max_lambda / result[i]))
        result[i] = safe_always_positive_division(max_lambda, (max_lambda - result[i]))

    return result

cpdef np.ndarray[np.float64_t, ndim=1] outlier_membership_vector(neighbor,
            lambda_, clusters, tree, max_lambda_dict, cluster_tree,
            softmax=False):

    cdef np.ndarray[np.float64_t, ndim=1] result

    if softmax:
        result = per_cluster_scores(neighbor, lambda_, clusters, tree,
                                    max_lambda_dict, cluster_tree)
        # Scale for numerical stability, mathematically equivalent with old
        # version due to the scaling with the sum in below.
        result = np.exp(result - np.nanmax(result))
        #result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = per_cluster_scores(neighbor, lambda_, clusters, tree,
                                    max_lambda_dict, cluster_tree)

    result /= result.sum()
    return result

cpdef np.float64_t prob_in_some_cluster(
        np.intp_t neighbor,
        np.float64_t lambda_,
        np.ndarray[np.intp_t, ndim=1] clusters,
        np.ndarray tree,
        dict max_lambda_dict,
        np.ndarray cluster_tree):
    """
    Compute the probability of a single point being in some cluster, considering
    the maximum height and resolving ties by selecting the cluster with the
    largest max_lambda value.

    Parameters:
    ----------
    neighbor : int
        The index of the point (neighbor) in the tree.
    lambda_ : float
        The lambda value associated with the point.
    clusters : np.ndarray[np.intp_t, ndim=1]
        Array of cluster labels.
    tree : np.ndarray
        Hierarchical tree structure.
    max_lambda_dict : dict
        Dictionary mapping cluster labels to their maximum lambda values.
    cluster_tree : np.ndarray
        Cluster tree structure.

    Returns:
    -------
    float
        The probability of the point being in some cluster.
    """
    cdef np.ndarray[np.float64_t, ndim=1] cluster_merge_heights
    cdef np.intp_t point_cluster
    cdef np.float64_t point_lambda
    cdef np.float64_t max_lambda
    cdef np.float64_t point_height
    cdef list candidate_indices
    cdef int selected_idx
    cdef int nearest_cluster
    cdef int i

    # Get the row in the tree corresponding to the neighbor
    point_row = get_tree_row_with_child(tree, neighbor)
    point_cluster = point_row['parent']
    point_lambda = lambda_

    # Compute merge heights
    cluster_merge_heights = merge_height(point_cluster, point_lambda, clusters, cluster_tree)
    point_height = cluster_merge_heights.max()

    # Find all indices where merge height == point_height
    candidate_indices = []
    for i in range(cluster_merge_heights.shape[0]):
        if cluster_merge_heights[i] == point_height:
            candidate_indices.append(i)

    # Among the candidates, pick the one with the maximum max_lambda_dict value
    selected_idx = candidate_indices[0]
    if len(candidate_indices) > 1:  # Only compare if there are multiple candidates
        for i in candidate_indices[1:]:
            if max_lambda_dict[clusters[i]] > max_lambda_dict[clusters[selected_idx]]:
                selected_idx = i

    # Get the nearest cluster and compute max_lambda
    nearest_cluster = clusters[selected_idx]
    max_lambda = max(lambda_, max_lambda_dict[nearest_cluster]) + 1e-8  # Avoid divide by zero

    # Compute and return the probability
    return point_height / max_lambda

cpdef np.ndarray[np.float64_t, ndim=2] all_points_per_cluster_scores(
        np.ndarray[np.intp_t, ndim=1] clusters,
        np.ndarray tree,
        dict max_lambda_dict,
        np.ndarray cluster_tree):

    cdef np.intp_t num_points = tree['parent'].min()
    cdef np.ndarray[np.float64_t, ndim=2] result_arr
    cdef np.float64_t[:, ::1] result
    cdef np.intp_t point
    cdef np.intp_t point_cluster
    cdef np.float64_t point_lambda
    cdef np.float64_t max_lambda

    cdef np.intp_t i, j

    result_arr = np.empty((num_points, clusters.shape[0]), dtype=np.float64)
    result = (<np.float64_t [:num_points, :clusters.shape[0]:1]>
                 (<np.float64_t *> result_arr.data))

    point_tree = tree[tree['child_size'] == 1]

    for i in range(point_tree.shape[0]):
        point_row = point_tree[i]
        point = point_row['child']
        point_cluster = point_row['parent']
        point_lambda = point_row['lambda_val']
        max_lambda = max_lambda_dict[point_cluster] + 1e-8 # avoid zero lambda

        # Can we not do a faster merge height operation here?
        result_arr[point] = merge_height(point_cluster, point_lambda,
                                          clusters, cluster_tree)

        # Cythonize: result = np.exp(-(max_lambda / height))

        ################################ Changes for new hdbscan probabilities ##############################
        # Cythonize: result = (max_lambda / (max_lambda - height))
        for j in range(result_arr.shape[1]):
            # result[point][j] = exp(-(max_lambda / result[point][j]))
            result[point][j] = max_lambda / (max_lambda - result[point][j])

        ######################################################################################################
    return result_arr

cpdef np.ndarray[np.float64_t, ndim=2] all_points_outlier_membership_vector(
        np.ndarray[np.intp_t, ndim=1] clusters,
        np.ndarray tree,
        dict max_lambda_dict,
        np.ndarray cluster_tree,
        np.intp_t softmax=False):

    cdef np.ndarray[np.float64_t, ndim=2] per_cluster_scores

    per_cluster_scores = all_points_per_cluster_scores(
                                clusters,
                                tree,
                                max_lambda_dict,
                                cluster_tree)
    if softmax:
        # Scale for numerical stability, mathematically equivalent with old
        # version due to the scaling with the sum in below.
        result = np.exp(per_cluster_scores - np.nanmax(per_cluster_scores))
        #result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = per_cluster_scores

    row_sums = result.sum(axis=1)
    result = result / row_sums[:, np.newaxis]

    return result

cpdef all_points_prob_in_some_cluster(
        np.ndarray[np.intp_t, ndim=1] clusters,
        np.ndarray tree,
        dict max_lambda_dict,
        np.ndarray cluster_tree):
    """
    Compute the probability of each point being in some cluster, considering
    the maximum height and resolving ties by selecting the cluster with the
    largest max_lambda value.

    Parameters:
    ----------
    clusters : np.ndarray[np.intp_t, ndim=1]
        Array of cluster labels.
    tree : np.ndarray
        Hierarchical tree structure.
    max_lambda_dict : dict
        Dictionary mapping cluster labels to their maximum lambda values.
    cluster_tree : np.ndarray
        Cluster tree structure.

    Returns:
    -------
    np.ndarray[np.float64_t, ndim=1]
        Array of probabilities for each point.
    """
    # Declare all variables at the top
    cdef np.ndarray[np.float64_t, ndim=1] heights
    cdef np.intp_t num_points = tree['parent'].min()
    cdef np.ndarray[np.float64_t, ndim=1] result
    cdef np.intp_t point
    cdef np.intp_t point_cluster
    cdef np.float64_t point_lambda
    cdef np.float64_t max_lambda
    cdef np.float64_t point_height
    cdef np.intp_t i, j
    cdef list candidate_indices
    cdef int selected_idx
    cdef int nearest_cluster

    # Initialize result array
    result = np.empty(num_points, dtype=np.float64)

    # Filter leaf nodes (points) from the tree
    point_tree = tree[tree['child_size'] == 1]

    for i in range(point_tree.shape[0]):
        point_row = point_tree[i]
        point = point_row['child']
        point_cluster = point_row['parent']
        point_lambda = point_row['lambda_val']

        # Compute merge heights
        heights = merge_height(point_cluster, point_lambda,
                               clusters, cluster_tree)
        point_height = heights.max()

        # Find all indices where the height equals the maximum height
        candidate_indices = []
        for j in range(heights.shape[0]):
            if heights[j] == point_height:
                candidate_indices.append(j)

        # Among the candidates, pick the one with the maximum max_lambda value
        selected_idx = candidate_indices[0]
        if len(candidate_indices) > 1:  # Only compare if there are multiple candidates
            for j in candidate_indices[1:]:
                if max_lambda_dict[clusters[j]] > max_lambda_dict[clusters[selected_idx]]:
                    selected_idx = j

        # Get the nearest cluster and compute max_lambda
        nearest_cluster = clusters[selected_idx]
        max_lambda = max(point_lambda, max_lambda_dict[nearest_cluster]) + 1e-8  # Avoid divide by zero

        # Compute the probability for the point
        result[point] = point_height / max_lambda

    return result


cpdef np.ndarray[np.float64_t, ndim=2] compute_merge_heights(
        np.ndarray[np.intp_t, ndim=1] clusters,
        np.ndarray tree,
        np.ndarray cluster_tree):
    """
    Compute the merge heights for each cluster.

    Parameters:
    ----------
    clusters : np.ndarray[np.intp_t, ndim=1]
        Array of cluster labels.
    tree : np.ndarray
        Hierarchical tree structure.
    cluster_tree : np.ndarray
        Cluster tree structure.

    Returns:
    -------
    np.ndarray[np.float64_t, ndim=2]
        Merge heights for each cluster.
    """
    cdef np.intp_t num_points = tree['parent'].min()
    cdef np.ndarray[np.float64_t, ndim=2] merge_heights = np.empty((num_points, clusters.shape[0]), dtype=np.float64)
    cdef np.intp_t i
    cdef np.intp_t point
    cdef np.intp_t point_cluster
    cdef np.float64_t point_lambda

    point_tree = tree[tree['child_size'] == 1]

    for i in range(point_tree.shape[0]):
        point_row = point_tree[i]
        point = point_row['child']
        point_cluster = point_row['parent']
        point_lambda = point_row['lambda_val']

        # Compute merge heights for the current point
        merge_heights[point] = merge_height(point_cluster, point_lambda, clusters, cluster_tree)

    return merge_heights