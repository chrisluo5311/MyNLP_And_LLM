import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

if __name__ == '__main__':
    # a = np.array([1,3,5])
    # b = np.array([0,-1,2])
    # print(euclidean_distance(a, b))

    yprob_test = np.loadtxt('../predict_test/yprob_test.txt')
    print(yprob_test)
    print(yprob_test.shape)
    print(type(yprob_test))