API Reference
=============

Major classes are :class:`HDBSCAN` and :class:`RobustSingleLinkage`.

HDBSCAN
-------

.. autoclass:: hdbscan.hdbscan_.HDBSCAN
   :members:

RobustSingleLinkage
-------------------

.. autoclass:: hdbscan.robust_single_linkage_.RobustSingleLinkage
   :members:


Utilities
---------

Other useful classes are contained in the plots module, the validity module,
and the prediction module.

.. autoclass:: hdbscan.plots.CondensedTree
   :members:

.. autoclass:: hdbscan.plots.SingleLinkageTree
   :members:

.. autoclass:: hdbscan.plots.MinimumSpanningTree
   :members:

.. automodule:: hdbscan.validity
   :members:

.. automodule:: hdbscan.prediction
   :members:


Branch detection
----------------

The branches module contains classes for detecting branches within clusters.

.. automodule:: hdbscan.branches
   :members: BranchDetector, detect_branches_in_clusters, approximate_predict_branch

.. autoclass:: hdbscan.plots.ApproximationGraph
   :members:
