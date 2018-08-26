import connecting_points as cp
import numpy as np


def test_einsum():
    a = np.arange(25).reshape(5, 5)
#    print(f'a:\n{a}')

    ans = np.einsum('ii', a)
#    print(f'Answer:\n{ans}')

    b = np.arange(5)
    ans = np.einsum('ij,j', a, b)
#    print(f'A:\n{a}\nB:\n{b}\nAns:\n{ans}')

    c = np.arange(6).reshape(2, 3)
    ans = np.einsum('ji', c)
#    print(f'C:\n{c}\nAns:\n{ans}')

    a = np.arange(10).reshape(5, 2)
    b = np.transpose(a)
    #b = np.arange(25).reshape(5, 5)
    ans = np.einsum('ij,ji->ij', a, b)
    print(f'A:\n{a}\nB:\n{b}\nAns:\n{ans}')


if __name__ == '__main__':
    test_einsum()
