# Uses python3
import sys
import numpy as np


class Node:
    def __init__(self, num):
        self.num = num
        self.location = None

    def __repr__(self):
        return 'Node({})'.format(self.num)


def convert_to_np_points(x, y):
    points = np.array([x, y]).transpose()
    return points


def calculate_distances(points):
    # Equation of distance between two points
    # (x^2-y^2) = x^2 + y^2 - 2*x*y

    return np.sqrt(np.einsum('ijk->ij', (points[:, None, :] - points)**2))


def ravel_distances(distances):
    indices = np.array(np.triu_indices(len(distances), 1)
                       ).transpose()
    raveled = [[ind[0], ind[1], distances[ind[0], ind[1]]] for ind in indices]
    return np.array(raveled)


def sort_distances(ravel_dist):
    sort_dist_inds = np.lexsort((ravel_dist[:, 0], ravel_dist[:, 2]))

    return ravel_dist[sort_dist_inds, :]


def is_valid_connection(node):
    if node.location:
        return False
    else:
        return True


def calc_min_distance(nodes, distances):
    first_node = nodes[int(distances[0][0])]
    used = set([first_node])
    first_node.location = first_node
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
        elif is_valid_connection(nodes[froms]):
            nodes[froms].location = nodes[tos]
            used.add(nodes[froms])
            total_distance += dist

        i += 1
    return total_distance


def minimum_distance(x, y):
    nodes = [Node(i) for i in range(len(x))]
    np_points = convert_to_np_points(x, y)
    distances = sort_distances(ravel_distances(calculate_distances(np_points)))
    result = calc_min_distance(nodes, distances)

    # write your code here
    return result


if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    x = data[1::2]
    y = data[2::2]
    print("{0:.9f}".format(minimum_distance(x, y)))
