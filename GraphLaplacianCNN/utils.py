# Class: MA398
# Author: Carson Crovo
# Date: 4-3-23

import os
import cv2
import warnings
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg.eigen
from itertools import repeat
from GraphLaplacianCNN import config

# Create Demo code
# Look closer at gcn framework at training vs. learning stage to find how results are combined.


class Calculate:
    # @staticmethod
    # def scaled_laplacian(frames: np.ndarray[np.ndarray[np.ndarray[np.float32]]]) -> spr.csr_matrix[np.float32]:
    #     laplacian = Calculate.graph_laplacian(frames)
    #     max_eigen, max_vect = spr.linalg.eigen.eigsh(laplacian, k=1, which='LM')
    #     return (2. / max_eigen[0]) * laplacian - spr.eye(laplacian.shape[0])

    @staticmethod
    def get_x_eigen(matrix: spr.csr_matrix, tolerance: float) ->\
            tuple[np.ndarray[np.float32], np.ndarray[np.ndarray[np.float32]], float]:
        all_eigen, all_vect = scipy.linalg.eigh(matrix.toarray())
        if tolerance <= 0: return all_eigen, all_vect, 0
        all_vect = all_vect.T
        first = 0
        last = all_eigen.size
        while first < last:
            mid = (first + last) // 2
            error2 = spr.linalg.norm(matrix - Calculate.restore_laplacian(all_eigen[(mid + 1):], all_vect[(mid + 1):]))
            error1 = spr.linalg.norm(matrix - Calculate.restore_laplacian(all_eigen[mid:], all_vect[mid:]))
            # print("Num vals:", all_eigen.size - mid)
            # print("Error1 is", error1, "Error2 is", error2)
            if error1 > tolerance: last = mid
            elif error2 <= tolerance: first = mid + 1
            else: return all_eigen[mid:], all_vect[mid:], error1
        return np.array([]), np.array([]), tolerance

    @staticmethod
    def restore_laplacian(eigenvalues: np.ndarray[np.float32], eigenvectors: np.ndarray[np.ndarray[np.float32]])\
            -> spr.csr_matrix:
        n = eigenvectors.shape[1]
        laplacian = np.zeros((n, n))
        for (val, vect) in zip(eigenvalues, eigenvectors):
            laplacian += val * np.outer(vect.T, vect)
        return spr.csr_matrix(laplacian, dtype=np.float32)

    @staticmethod
    def graph_laplacian(frames: np.ndarray[np.ndarray[np.ndarray[np.float32]]]) -> spr.csr_matrix:
        adj_matrix =\
            Calculate._temporal_splice(frames, config.SIGMA2) if config.SPLICE == 'temporal' else\
            Calculate._spatial_splice(frames, config.K, config.SIGMA2)
        return Calculate.adj_to_laplacian(adj_matrix, config.LAPLACIAN_METHOD)

    @staticmethod
    def adj_to_laplacian(adj_matrix: spr.csr_matrix, method: str) -> spr.csr_matrix:
        sums = adj_matrix.sum(axis=1, dtype=float).ravel()
        deg_matrix = spr.spdiags(sums, 0, sums.size, sums.size)
        L = deg_matrix - adj_matrix
        if method == 'unnormalized':
            return L
        elif method == 'symmetric':
            d_inverse_sqrt = spr.spdiags(np.power(sums, -0.5), 0, sums.size, sums.size)
            return (d_inverse_sqrt @ L) @ d_inverse_sqrt
        elif method == 'random':
            d_inverse = spr.spdiags(np.power(sums, -1), 0, sums.size, sums.size)
            return d_inverse @ L
        raise Exception("Invalid 'method' parameter provided for method graph_laplacian.")

    @staticmethod
    def _temporal_splice(frames: np.ndarray[np.ndarray[np.ndarray[np.float32]]], sigma2: float)\
            -> spr.csr_matrix:
        if sigma2 <= 50000: warnings.showwarning("'sigma2' parameter may be too low.", UserWarning, 'utils.py', 40)

        m = len(frames)  # num frames
        n = frames[0].size  # num pixels
        MxN = np.zeros((m, n))
        for i in range(m): MxN[i] = frames[i].ravel()  # concatenate rows of each frame to form a mxn matrix
        dist = scipy.spatial.distance.pdist(MxN, 'sqeuclidean')
        return spr.csr_matrix(scipy.spatial.distance.squareform(
            np.exp(  # then use pdist and exp() formula with sigma to find adj matrix
                -dist / sigma2  # max(dist)
            )
        ), dtype=np.float32)

    @staticmethod
    def _spatial_splice(frames: np.ndarray[np.ndarray[np.ndarray[np.float32]]], k: int, sigma2: float):
        if sigma2 <= 50000: warnings.showwarning("'sigma2' parameter may be too low.", UserWarning, 'utils.py', 56)
        if k <= 0: raise Exception("'k' parameter was negative.")

        # converts frames in 3d: n x m1 x m2 into voxels in 2d: (m1 * m2) x n (or m x n)
        # total is 76800 x 410
        voxels = np.array(frames).reshape(len(frames), -1).T

        # get index matrix representing k nearest neighbors size m x k for m voxels and k neighbors to each voxel
        index_matrix = Calculate._k_nearest(frames[0].shape, k)

        # neighbor_values is same m x k format as index_matrix, but uses the actual frame values instead of indices
        # neighbor_values = voxels[index_matrix, :]

        # Assume above formula results in 0 if Vi and Vj are not neighbors:
        # Default adjacency matrix is 0s
        adj = []
        # Iterate over all 'neighboring' pairs of indices
        for i, neighbors in enumerate(index_matrix):
            neighbors = np.unique(neighbors)
            for j in neighbors:  # for each neighboring pair:
                if j < i and i in index_matrix[j]: continue  # skip calculating what has already been calculated
                # For adjacency matrix (using patch) use ||N_p(Vi)-N_p(Vj)||_F^2 (Frobenius norm)
                val = np.sum((voxels[i] - voxels[j]) ** 2)
                val = np.exp(-val / sigma2)
                adj.append([val, i, j])
                adj.append([val, j, i])
        adj = np.array(adj)
        return spr.csr_matrix((adj[:, 0], (adj[:, 1], adj[:, 2])), shape=(len(voxels), len(voxels)), dtype=np.float32)

    @staticmethod
    # This method calculates the 'k' nearest neighboring indices for each index in a 2d matrix with 'shape' dimensions
    def _k_nearest(shape: tuple[int, int], k: int) -> np.ndarray[np.ndarray[int]]:
        # Approximate ray-casting radius based on k
        radius = int(np.ceil(np.sqrt((k + 1) / np.pi))) + 1

        # Make a padded matrix (extend frame size) to help with geometric flipping
        # during this process, extend border pixels geometrically and copy the cardinal NSEW indices across x or y-axis
        pad_array = np.pad(np.arange(shape[0] * shape[1]).reshape(shape), radius, mode='reflect')
        pad_width = shape[1] + 2 * radius

        # Calculate a circular ray-cast to apply to indices to get k nearest neighbors
        dim = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(dim, dim)
        distances = np.sqrt(X ** 2 + Y ** 2)
        mask = (distances <= radius) & (distances != 0)
        ray_cast = X[mask] + Y[mask] * pad_width
        ray_cast = ray_cast[np.argsort(distances[mask])[:k]]

        # Get the indices of the frame after padding applied
        ind_array = np.arange(pad_array.size).reshape(pad_array.shape)[radius:-radius, radius:-radius]
        # Apply the ray-cast to each index on the frame
        indices = map(Calculate.__neighbors, ind_array.ravel(), repeat(ray_cast), repeat(pad_array.ravel()))
        # Return the n x k ndarray representing k neighbors for each of n indices
        return np.array(list(indices))

    @staticmethod
    # This is a private method only meant to be used within the 'map()' call within k_nearest method
    def __neighbors(index: int, ray_cast: np.ndarray[int], neighborhood: np.ndarray[int]) -> np.ndarray[int]:
        return neighborhood[index + ray_cast]  # returns a 1d array after applying ray-cast to a single index


# Read 410 .JPGs into 240 x 320 matrices of gray pixels represented by integers (or RGB integer vectors of size 3)
def read_video(directory: str, resize_pics: tuple[int, int] = None) -> np.ndarray[np.ndarray[np.ndarray[np.float32]]]:
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            raise Exception("File \"", filepath, "\" not found in directory \"", directory, "\".")
        image = cv2.imread(filepath)
        if resize_pics is not None: image = cv2.resize(image, resize_pics, 0, 0, cv2.INTER_LINEAR)
        image = np.multiply(image.astype(np.float32), 1.0 / 255.0)
        images.append(image)
    images = np.array(images)
    return images
