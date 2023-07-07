# https://docs.python.org/3/library/unittest.html

import unittest
import numpy.testing
import numpy as np
import scipy.sparse as spr
from utils import Calculate

if __name__ == '__main__':
    unittest.main()


class TestCalculations(unittest.TestCase):
    def test_k_nearest(self):
        self.assertEqual([[0] * 10], Calculate._k_nearest((1, 1), 10).tolist())
        self.assertEqual([[0]], Calculate._k_nearest((1, 1), 1).tolist())
        self.assertEqual((63, 5), Calculate._k_nearest((9, 7), 5).shape)
        self.assertEqual([[1, 3, 3],
                          [0, 4, 4],
                          [1, 5, 5],
                          [4, 0, 0],
                          [3, 1, 1],
                          [4, 2, 2]],
                         Calculate._k_nearest((2, 3), 3).tolist()
                         )
        with self.assertRaisesRegex(ValueError, "'k' parameter must be positive."):
            Calculate._k_nearest((1, 1), k=0)
            Calculate._k_nearest((1, 1), k=-5)

    def test_spatial_splice(self):
        e_1 = np.exp(-1)
        e_4 = np.exp(-4)
        with self.assertWarnsRegex(UserWarning, "'sigma2' parameter may be too low."):
            np.testing.assert_allclose(
                np.array([[0, e_1, 0, e_1, 0, 0, 0, 0, 0],
                          [e_1, 0, e_4, 0, e_4, 0, 0, 0, 0],
                          [0, e_4, 0, 0, 0, 1, 0, 0, 0],
                          [e_1, 0, 0, 0, e_4, 0, e_4, 0, 0],
                          [0, e_4, 0, e_4, 0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, e_4, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 1, 0]]),
                Calculate._spatial_splice(np.array([
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]]
                ]), k=4, sigma2=50000).todense()
            )

    def test_temporal_splice(self):
        self.assertEqual((6, 6), Calculate._temporal_splice(np.array([
            [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
            [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
            [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
            [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
            [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
            [[100, 200, 0], [200, 0, 0], [0, 0, 0]]
        ]), 50001).shape)
        with self.assertWarnsRegex(UserWarning, "'sigma2' parameter may be too low."):
            np.testing.assert_allclose(
                1 - np.eye(5),
                Calculate._temporal_splice(np.array([
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                    [[100, 200, 0], [200, 0, 0], [0, 0, 0]],
                ]), 50000).todense())
        e_1 = np.exp(-1)
        e_4 = np.exp(-4)
        np.testing.assert_allclose(
            np.array([
                [0, e_1, e_4],
                [e_1, 0, e_1],
                [e_4, e_1, 0]
            ]),
            Calculate._temporal_splice(np.array([
                [[100, 100, 100], [100, 100, 100], [100, 100, 100]],
                [[200, 200, 200], [200, 200, 200], [200, 200, 200]],
                [[300, 300, 300], [300, 300, 300], [300, 300, 300]]
            ]), 90000).todense())

    def test_adj_to_laplacian(self):
        np.testing.assert_allclose(
            np.array([
                [-6, 1, 2, 3],
                [1, -1, 0, 0],
                [2, 0, -6, 4],
                [3, 0, 4, -7]
            ]),
            Calculate.adj_to_laplacian(
                spr.csr_matrix(np.array([
                    [6, -1, -2, -3],
                    [-1, 1, 0, 0],
                    [-2, 0, 6, -4],
                    [-3, 0, -4, 7]
                ])), method='unnormalized').todense()
        )
        np.testing.assert_allclose(
            np.array([
                [1, 0, -1 / 8, -3 / 10],  # unnormalized / 2
                [0, 1, -1 / 12, -8 / 15],  # unnormalized / 3
                [-1 / 8, -1 / 12, 1, -14 / 20],  # unnormalized / 4
                [-3 / 10, -8 / 15, -14 / 20, 1]  # unnormalized / 5
                # /2      /3      /4      /5
            ]),
            Calculate.adj_to_laplacian(
                spr.csr_matrix(np.array([
                    [0, 0, 1, 3],  # row sum 4
                    [0, 0, 1, 8],  # row sum 9
                    [1, 1, 0, 14],  # row sum 16
                    [3, 8, 14, 0]  # row sum 25
                ])), method='symmetric').todense()
        )
        np.testing.assert_allclose(
            np.array([
                [1, 0, -1 / 4, -3 / 4],  # unnormalized / 4
                [0, 1, -1 / 9, -8 / 9],  # unnormalized / 9
                [-1 / 16, -1 / 16, 1, -14 / 16],  # unnormalized / 16
                [-3 / 25, -8 / 25, -14 / 25, 1]  # unnormalized / 25
            ]),
            Calculate.adj_to_laplacian(
                spr.csr_matrix(np.array([
                    [0, 0, 1, 3],  # row sum 4
                    [0, 0, 1, 8],  # row sum 9
                    [1, 1, 0, 14],  # row sum 16
                    [3, 8, 14, 0]  # row sum 25
                ])), method='random').todense()
        )
        with self.assertRaisesRegex(ValueError, "Invalid 'method' parameter provided"):
            Calculate.adj_to_laplacian(spr.csr_matrix([[0]]), method='incorrect_argument')

    def test_restore_laplacian(self):
        r05 = np.sqrt(0.5)
        r2 = np.sqrt(2)
        self.assertAlmostEqual(
            0,
            np.linalg.norm(
                np.array([
                    [1, -r05, 0],
                    [-r05, 1, -r05],
                    [0, -r05, 1]
                ], dtype=np.float32)
                - Calculate.restore_laplacian(
                    eigenvalues=np.array([2, 1, 0]),
                    eigenvectors=np.array([
                        [1 / 2, -r2 / 2, 1 / 2],
                        [-1 / r2, 0, 1 / r2],
                        [1 / 2, r2 / 2, 1 / 2]
                    ])
                ).todense()
            ),
            places=12
        )

    # def test_get_x_eigen(self):
    #     pass
