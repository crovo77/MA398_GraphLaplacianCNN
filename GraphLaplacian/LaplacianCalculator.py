# Class: MA398
# Author: Carson Crovo
# Date: 4-3-23

import os
import numpy as np
import scipy as sp
import cv2
from itertools import repeat


class LaplacianCalculator:
    def __init__(self, directory: str = None, frames: np.ndarray[np.ndarray[np.ndarray[np.float32]]] = None):
        if frames is not None:
            self._frames = frames
        elif directory is not None:
            self._frames = LaplacianCalculator.read_video(directory)
        else: raise Exception("LaplacianCalculator class constructor requires either a directory or frames.")

    def calc_graph_laplacian(self, splice: str, method: str, k: int = 0, sigma2: float = 0) -> sp.sparse.csr_matrix:
        if splice == 'spatial' and k > 0 and sigma2 > 0:
            return LaplacianCalculator.graph_laplacian(
                LaplacianCalculator.spatial_splice(self._frames, k, sigma2), method
            )
        elif splice == 'temporal' and sigma2 > 0:
            return LaplacianCalculator.graph_laplacian(
                LaplacianCalculator.temporal_splice(self._frames, sigma2), method
            )
        raise Exception("Invalid argument(s) provided for calc_graph_laplacian()")

    @staticmethod
    def graph_laplacian(adj_matrix: sp.sparse.csr_matrix, method: str) -> sp.sparse.csr_matrix:
        sums = adj_matrix.sum(axis=1, dtype=float).ravel()
        deg_matrix = sp.sparse.spdiags(sums, 0, sums.size, sums.size)
        L = deg_matrix - adj_matrix
        if method == 'unnormalized':
            return L
        elif method == 'symmetric':
            d_inverse_sqrt = sp.sparse.spdiags(np.power(sums, -0.5), 0, sums.size, sums.size)
            return (d_inverse_sqrt @ L) @ d_inverse_sqrt
        elif method == 'random':
            d_inverse = sp.sparse.spdiags(np.power(sums, -1), 0, sums.size, sums.size)
            return d_inverse @ L
        else: raise Exception("Invalid arguments provided for calculating graph_laplacian.")

    @staticmethod
    def read_video(directory: str, resize: tuple[int, int] = None) -> np.ndarray[np.ndarray[np.ndarray[np.float32]]]:
        images = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath):
                raise Exception("File \"", filepath, "\" not found in directory \"", directory, "\".")
            image = cv2.imread(filepath)
            if resize is not None: image = cv2.resize(image, resize, 0, 0, cv2.INTER_LINEAR)
            image = np.multiply(image.astype(np.float32), 1.0 / 255.0)
            images.append(image)
        images = np.array(images)
        return images

    # Temporal Method:
    @staticmethod
    def temporal_splice(frames: np.ndarray[np.ndarray[np.ndarray[np.float32]]], sigma2: float)\
            -> sp.sparse.csr_matrix[np.float32]:
        m = len(frames)  # num frames
        n = frames[0].size  # num pixels
        MxN = np.zeros((m, n))
        for i in range(m): MxN[i] = frames[i].ravel()  # concatenate rows of each frame to form a mxn matrix
        dist = sp.spatial.distance.pdist(MxN, 'sqeuclidean')
        return sp.sparse.csr_matrix(sp.spatial.distance.squareform(
            np.exp(  # then use pdist and exp() formula with sigma to find adj matrix
                -dist / sigma2  # max(dist)
            )
        ), dtype=np.float32)

    @staticmethod
    def spatial_splice(frames: np.ndarray[np.ndarray[np.ndarray[np.float32]]], k: int, sigma2: float)\
            -> sp.sparse.csr_matrix[np.float32]:
        # converts frames in 3d: n x m1 x m2 into voxels in 2d: (m1 * m2) x n (or m x n)
        # total is 76800 x 410
        voxels = np.array(frames).reshape(len(frames), -1).T
        # get index matrix representing k nearest neighbors size m x k for m voxels and k neighbors to each voxel
        index_matrix = LaplacianCalculator.k_nearest(frames[0].shape, k)

        # neighbor_values is same m x k format as index_matrix, but uses the actual frame values instead of indices
        neighbor_values = voxels[index_matrix, :]

        # Assume above formula results in 0 if Vi and Vj are not neighbors:
        # Default adjacency matrix is 0s
        adj = []
        # Iterate over all 'neighboring' pairs of indices
        for i, neighbors in enumerate(index_matrix):
            for j in neighbors:  # for each neighboring pair:
                if j < i and i in index_matrix[j]: continue  # skip calculating what has already been calculated
                # For adjacency matrix (using patch) use ||N_p(Vi)-N_p(Vj)||_F^2 (Frobenius norm)
                val = np.sum((neighbor_values[i] - neighbor_values[j]) ** 2)
                val = np.exp(-val / sigma2)
                adj.append([val, i, j])
                adj.append([val, j, i])
        if not adj: raise Exception("Incorrect arguments provided for calculating spatial_splice.")
        adj = np.array(adj)
        return sp.sparse.csr_matrix((adj[:, 0], (adj[:, 1], adj[:, 2])), shape=(len(voxels), len(voxels)))

    @staticmethod
    # This method calculates the 'k' nearest neighboring indices for each index in a 2d matrix with 'shape' dimensions
    def k_nearest(shape: tuple[int, int], k: int) -> np.ndarray[np.ndarray]:
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
        indices = map(LaplacianCalculator.__neighbors, ind_array.ravel(), repeat(ray_cast), repeat(pad_array.ravel()))
        # Return the n x k ndarray representing k neighbors for each of n indices
        return np.array(list(indices))

    @staticmethod
    # This is a private method only meant to be used within the 'map()' call within k_nearest method
    def __neighbors(index: int, ray_cast: np.ndarray[int], neighborhood: np.ndarray[int]) -> np.ndarray[int]:
        return neighborhood[index + ray_cast]  # returns a 1d array after applying ray-cast to a single index


def main():
    # Read 410 .JPGs into 240 x 320 matrices of gray pixels represented by integers (or RGB integer vectors of size 3)
    video = LaplacianCalculator(directory='./RawData/')
    arr = video.calc_graph_laplacian(splice='temporal', k=8, sigma2=100000, method='random')
    print('arr shape is', arr.shape)
    print('arr is', arr)


if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
