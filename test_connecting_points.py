import connecting_points as cp
import numpy as np
import pytest


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
    print(f'Ans:\n{ans}\nDist:\n{dist}')
    assert dist == pytest.approx(ans)

    x = [0, 1, 2]
    y = [0, 1, 2]
    points = cp.convert_to_np_points(x, y)
    ans = calc_dist_slow(points)
    dist = cp.calculate_distances(points)
    print(f'Ans:\n{ans}\nDist:\n{dist}')
    assert dist == pytest.approx(ans)

    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    points = cp.convert_to_np_points(x, y)
    ans = calc_dist_slow(points)
    dist = cp.calculate_distances(points)
    print(f'Ans:\n{ans}\nDist:\n{dist}')
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


def test_node():
    node = cp.Node(0)
    assert node
    assert node.num == 0
    assert node.location == node


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


if __name__ == '__main__':
    #   test_distance()
    #    test_convert_to_points()
    test_sort_points()
    #    test_minumum_distance()
