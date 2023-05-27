# Class: MA398
# Author: Carson Crovo
# Date: 5-27-23

# 'SPLICE' determines how a video (3D matrix) is converted into an adjacency matrix
# 'SPLICE' may be set to either 'temporal' or 'spatial'
SPLICE = 'temporal'

# 'K' is a positive integer used during a spatial splice in a radial k-nearest neighbor algorithm to find clusters
# 'K' works best when set to integers divisible by 4
K = 8

# 'SIGMA2' (or sigma squared) is variance used within splice methods when finding adjacency matrix
# 'SIGMA2' should be a large positive value (near 100000) to keep adjacency matrices smaller
SIGMA2: float = 100000.

# 'LAPLACIAN_METHOD' specifies the type of graph laplacian calculated
# 'LAPLACIAN_METHOD' may be set to 'unnormalized', 'symmetric', or 'random'
LAPLACIAN_METHOD = 'unnormalized'
