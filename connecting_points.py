# Uses python3
import sys
import numpy as np


class Node:
    def __init__(self, num):
        self.num = num
        self.location = self

    def __repr__(self):
        return 'Node({})'.format(self.num)


def convert_to_np_points(x, y):
    points = np.array((x, y)).reshape(len(x), 2)
    return points


def calculate_distances(points):
    # Equation of distance between two points
    # sqrt(x^2-y^2) = sqrt(x^2 + y^2 - 2*x*y)
    return np.sqrt((points**2).sum(axis=1)[:, None]
                   + ((points**2).sum(axis=1)[None, :])
                   - 2 * points.dot(points.transpose()))


def ravel_distances(distances):
    indices = np.array(np.triu_indices(len(distances), 1)
                       ).transpose()
    raveled = [[ind[0], ind[1], distances[ind[0], ind[1]]] for ind in indices]
    return np.array(raveled)


def sort_distances(ravel_dist):
    sort_dist = ravel_dist[np.lexsort(([1, 0, 1]*ravel_dist).T)]
    return sort_dist


def is_valid_connection(node):
    if node.location == node:
        return True
    else:
        return False


def minimum_distance(x, y):
    nodes = [Node(i) for i in range(len(x))]
    np_points = convert_to_np_points(x, y)
    distances = sort_distances(ravel_distances(calculate_distances(np_points)))
    used = set()
    total_distance = 0
    i = 0
    while len(used) < len(nodes):
        froms, tos, dist = distances[i]
        froms = int(froms)
        tos = int(tos)
        if is_valid_connection(nodes[tos]):
            nodes[tos].location = nodes[froms]
            used.add(nodes[tos])
            total_distance += dist
        i += 1

    # write your code here
    return total_distance


if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    x = data[1::2]
    y = data[2::2]
    print("{0:.9f}".format(minimum_distance(x, y)))
