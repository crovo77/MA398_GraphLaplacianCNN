# Class: MA398
# Author: Carson Crovo
# Date: 5-27-23
import scipy.sparse

from GraphLaplacianCNN import utils
import scipy.sparse.linalg as lg
import scipy.sparse as spr
import matplotlib.pyplot as plt
import numpy as np


def main():
    video = utils.read_video("./RawData")
    print("video length is", len(video))
    laplacian = utils.Calculate.graph_laplacian(video)
    # laplacian = np.array([[6, -1, -2, -3],
    #                       [-1, 1, 0, 0],
    #                       [-2, 0, 6, -4],
    #                       [-3, 0, -4, 7]], dtype=np.float32)
    # laplacian = spr.csr_matrix(laplacian)
    # print("laplacian shape is", laplacian.shape)
    # val, vect = lg.eigsh(laplacian.toarray(), k=2, which='LM')
    # vect = vect.T
    # print("val shape is", val.shape)
    # print("vect shape is", vect.shape)
    # print("vals are", val)
    # print("vect are", vect)
    #
    # plt.plot(val)
    # plt.show()

    val, vect, error = utils.Calculate.get_x_eigen(laplacian, tolerance=50)
    print(val.size)
    print(error)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
