# https://docs.python.org/3/library/unittest.html

import unittest
import numpy.testing
import numpy as np
from utils import Calculate

if __name__ == '__main__':
    unittest.main()


class TestCalculations(unittest.TestCase):
    def test_k_nearest(self):
        self.assertEqual(Calculate._k_nearest((1, 1), 10).tolist(), [[0] * 10])
        self.assertEqual(Calculate._k_nearest((1, 1), 0).tolist(), [[]])
        self.assertEqual(Calculate._k_nearest((9, 7), 5).shape, (63, 5))
        self.assertEqual([[1, 3, 3],
                          [0, 4, 4],
                          [1, 5, 5],
                          [4, 0, 0],
                          [3, 1, 1],
                          [4, 2, 2]],
                         Calculate._k_nearest((2, 3), 3).tolist()
                         )

    def test_spatial_splice(self):
        e_1 = np.exp(-1)
        e_4 = np.exp(-4)
        # with self.assertWarns(UserWarning):
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

    # def test_get_x_eigen(self):
    #     pass
