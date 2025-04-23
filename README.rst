For the full README of the original package, please visit `hdbscan <https://github.com/scikit-learn-contrib/hdbscan>`_

This forked repository deals with the soft clustering method described in `How Soft Clustering for HDBSCAN Works <https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html>`_ with a couple of bugs fixed and trying to address the issue `#628 <https://github.com/scikit-learn-contrib/hdbscan/issues/628>`_.

------------------------
Additional/Updated functions in this package
------------------------

1) **all_points_membership_vectors**: Combined soft clustering membership probability vectors for each overdensity
2) **all_points_outlier_vectors**: Outlier-based membership probabilities for each overdensity
3) **all_points_distance_vectors**: Distance-based membership probs for each overdensity
4) **all_points_prob_in_some_cluster_vectors**: Factor used to convert conditional to unconditional probability for each data point

These functions can be called in a similar fashion as in the original package: hdbscan.all_points_membership_vectors(clusterer). Please make sure that **prediction_data** is set to **True** when initialising *clusterer*

Based on the paper:
    R.J.G.B. Campello, D. Moulavi, A. Zimek and J. Sander 
    *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection*, 
    ACM Trans. on Knowledge Discovery from Data, Vol 10, 1 (July 2015), 1-51.


---------
Licensing
---------

The hdbscan package is 3-clause BSD licensed. Enjoy.
