import connecting_points as cp
import numpy as np
import pytest


def test_convert_to_points():
    x = [0, 1, 2, 3, 4, 5]
    y = [6, 7, 8, 9, 10, 11]
    points = cp.convert_to_np_points(x, y)
    ans = np.arange(12).reshape(6, 2)
    assert points.all() == ans.all()


def test_distance():
    x = [0, 1, 2]
    y = [0, 1, 2]
    points = cp.convert_to_np_points(x, y)
    ans = np.array([[0, np.sqrt(2), np.sqrt(4)],
                    [np.sqrt(2), 0, np.sqrt(2)],
                    [np.sqrt(4), np.sqrt(2), 0]])

    assert cp.calculate_distances(points).all() == ans.all()


def test_ravel_froms_tos_distances():
    x = [0, 1, 2]
    y = [0, 1, 2]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)

    answer = [[0, 1, np.sqrt(2)],
              [0, 2, np.sqrt(4)],
              [1, 2, np.sqrt(2)]]

    answer = np.array(answer)
    assert cp.ravel_distances(dist).all() == answer.all()


def test_sort_points():
    x = [0, 1, 2]
    y = [0, 1, 2]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)

    answer_1 = cp.sort_distances(cp.ravel_distances(dist))

    x = [1, 0, 2]
    y = [1, 0, 2]
    points = cp.convert_to_np_points(x, y)
    dist = cp.calculate_distances(points)

    answer_2 = cp.sort_distances(cp.ravel_distances(dist))
    print(answer_1)
    print(answer_2)
    assert answer_1.all() == answer_2.all()


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


if __name__ == '__main__':
    #    test_distance()
    #    test_convert_to_points()
    #    test_sort_points()
    test_minumum_distance()
