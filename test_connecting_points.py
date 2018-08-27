import connecting_points as cp
import numpy as np
import pytest
from random import randint
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd


def make_and_save_data():
    x_list = []
    y_list = []
    pairs = set()
    while len(pairs) < 20:
        x = randint(-1000, 1000)
        y = randint(-1000, 1000)
        if (x, y) not in pairs:
            pairs.add((x, y))
            x_list.append(x)
            y_list.append(y)

    excel = pd.ExcelWriter('data.xlsx')
    data = pd.DataFrame(list(zip(x_list, y_list)), columns=['x', 'y'])
    data.to_excel(excel, sheet_name='points')
    dist = cp.calculate_distances(data.values)
    dist_df = pd.DataFrame(np.tril(dist))
    dist_df.to_excel(excel, sheet_name='distances')
    sort_dist = pd.DataFrame(cp.sort_distances(cp.ravel_distances(dist)))
    sort_dist.to_excel(excel, sheet_name='sorted dist')

    csr = csr_matrix(np.triu(dist))
    csr_dict = csr.todok()
    csr_list = np.array([[nodes[0], nodes[1], distance]
                         for nodes, distance in csr_dict.items()])
    sort_csr = pd.DataFrame(cp.sort_distances(csr_list))
    sort_csr.to_excel(excel, sheet_name='sorted csr')

    path = minimum_spanning_tree(csr)
    path = pd.DataFrame(path.toarray())
    path.to_excel(excel, sheet_name='scipy path')

    excel.save()
    print('SAVED DATA')


def read_data():
    data = pd.read_excel('data.xlsx', indexcol=0,
                         header=0, sheet_name='points')
    x = data['x'].values
    y = data['y'].values
    return x, y


def calc_dist_slow(points):
    answer = []
    for i_1, j_1 in points:
        current_row = []
        for i_2, j_2 in points:
            distance = ((i_2-i_1)**2+(j_2-j_1)**2)**.5
            current_row.append(distance)
        answer.append(current_row)

    return np.array(answer)


def test_convert_to_points():
    x = [0, 1, 2, 3, 4, 5]
    y = [6, 7, 8, 9, 10, 11]
    points = cp.convert_to_np_points(x, y)
    ans = np.array([[0, 6],
                    [1, 7],
                    [2, 8],
                    [3, 9],
                    [4, 10],
                    [5, 11]])
    assert points == pytest.approx(ans)

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    points = cp.convert_to_np_points(x, y)
    ans = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

    assert points == pytest.approx(ans)


def test_distance():
    x = [0, 1]
    y = [1, 0]
    points = cp.convert_to_np_points(x, y)
    ans = calc_dist_slow(points)
    dist = cp.calculate_distances(points)
    assert dist == pytest.approx(ans)

    x = [0, 1, 2]
    y = [0, 1, 2]
    points = cp.convert_to_np_points(x, y)
    ans = calc_dist_slow(points)
    dist = cp.calculate_distances(points)
    assert dist == pytest.approx(ans)

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    points = cp.convert_to_np_points(x, y)
    ans = calc_dist_slow(points)
    dist = cp.calculate_distances(points)
    assert dist == pytest.approx(ans)

    x = [0, 0, 1, 3, 3]
    y = [0, 2, 1, 0, 2]
    points = cp.convert_to_np_points(x, y)
    ans = calc_dist_slow(points)
    dist = cp.calculate_distances(points)
    assert dist == pytest.approx(ans)


def test_ravel_distances():
    x = [0, 1, 2]
    y = [0, 1, 2]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)

    answer = [[0, 1, dist[0, 1]],
              [0, 2, dist[0, 2]],
              [1, 2, dist[1, 2]]]

    answer = np.array(answer)
    dist = cp.ravel_distances(dist)

    assert dist == pytest.approx(answer)

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)
    answer = np.array([[0, 1, dist[0, 1]],
                       [0, 2, dist[0, 2]],
                       [0, 3, dist[0, 3]],
                       [1, 2, dist[1, 2]],
                       [1, 3, dist[1, 3]],
                       [2, 3, dist[2, 3]]])

    dist = cp.ravel_distances(dist)

    assert dist == pytest.approx(answer)

    x = [0, 0, 1, 3, 3]
    y = [0, 2, 1, 0, 2]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)
    answer = np.array([[0, 1, dist[0, 1]],
                       [0, 2, dist[0, 2]],
                       [0, 3, dist[0, 3]],
                       [0, 4, dist[0, 4]],
                       [1, 2, dist[1, 2]],
                       [1, 3, dist[1, 3]],
                       [1, 4, dist[1, 4]],
                       [2, 3, dist[2, 3]],
                       [2, 4, dist[2, 4]],
                       [3, 4, dist[3, 4]]])
    dist = cp.ravel_distances(dist)
    assert dist == pytest.approx(answer)


def test_calc_min_distance():
    nodes = [cp.Node(i) for i in range(3)]


def test_sort_points():
    x = [0, 1, 2]
    y = [0, 1, 2]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)
    sort = cp.sort_distances(cp.ravel_distances(dist))
    answer = np.array([[0.,         1.,         1.41421356],
                       [1.,         2.,         1.41421356],
                       [0.,         2.,         2.82842712]])
    assert sort == pytest.approx(answer)

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    points = cp.convert_to_np_points(x, y)
    dist = cp.sort_distances(cp.ravel_distances(
        cp.calculate_distances(points)))
    answer = np.array([[0, 1, 1],
                       [0, 2, 1],
                       [1, 3, 1],
                       [2, 3, 1],
                       [0, 3, np.sqrt(2)],
                       [1, 2, np.sqrt(2)]])

    assert dist == pytest.approx(answer)

    x = [0, 0, 1, 3, 3]
    y = [0, 2, 1, 0, 2]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)
    dist = cp.ravel_distances(dist)
    dist = cp.sort_distances(dist)


def test_node():
    node = cp.Node(0)
    assert node
    assert node.num == 0
    assert not node.location


def test_is_valid_connection():

    node_1 = cp.Node(0)
    assert cp.is_valid_connection(node_1)

    node_2 = cp.Node(1)
    node_2.location = node_1
    assert not cp.is_valid_connection(node_2)


def test_minumum_distance():
    test_1_x = [0, 0, 1, 1]
    test_1_y = [0, 1, 0, 1]
    assert cp.minimum_distance(
        test_1_x, test_1_y) == pytest.approx(3.0, rel=1e-7)

    test_2_x = [0, 0, 1, 3, 3]
    test_2_y = [0, 2, 1, 0, 2]
    assert cp.minimum_distance(
        test_2_x, test_2_y) == pytest.approx(7.064495102, rel=1e-7)


def test_min_distance_speed():
    x_list, y_list = read_data()
    start = time.time()
    my_dist = cp.minimum_distance(x_list, y_list)
    finished = time.time()

    assert finished-start < 10
    points = cp.convert_to_np_points(x_list, y_list)
    dist = cp.calculate_distances(points)
    dist = csr_matrix(np.triu(dist))

    Tcsr = minimum_spanning_tree(dist)
    assert my_dist == pytest.approx(Tcsr.toarray().sum())


if __name__ == '__main__':
    make_and_save_data()

    #    test_distance()
    #    test_convert_to_points()
    #    test_ravel_distances()
    #    test_sort_points()
    #    test_minumum_distance()
#    test_min_distance_speed()
