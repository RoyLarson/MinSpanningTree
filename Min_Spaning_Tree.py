# Uses python3
import sys
import numpy as np


class Node:
    def __init__(self, num):
        self.num = num
        self.location = self
        self.connected_nodes = set([self])

    def __repr__(self):
        return 'Node({})'.format(self.num)

    def add_and_relocate_nodes(self, from_node):
        self.location.connected_nodes.update(
            from_node.location.connected_nodes)
        for node in from_node.location.connected_nodes:
            node.location = self.location


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


def connect_nodes(node_one, node_two):
    """
    connects nodes so that all of the nodes 
    will end up at Node(0)
    """
    if node_one.location.num < node_two.location.num:
        node_one.location.add_and_relocate_nodes(node_two)
    else:
        node_two.location.add_and_relocate_nodes(node_one)


def is_valid_connection(node_one, node_two):
    if node_two in node_one.location.connected_nodes:
        return False
    else:
        return True


def calc_min_distance(nodes, distances):
    final_target_node = nodes[0]
    total_distance = 0
    i = 0
    while len(final_target_node.connected_nodes) < len(nodes):
        node_one_i, node_two_i, dist = distances[i]
        node_one = nodes[int(node_one_i)]
        node_two = nodes[int(node_two_i)]

        if is_valid_connection(node_one, node_two):
            connect_nodes(node_one, node_two)
            total_distance += dist

        i += 1

    return total_distance


def minimum_distance(x, y):
    if len(x) == 1:
        return 0
    nodes = [Node(i) for i in range(len(x))]
    np_points = convert_to_np_points(x, y)
    distances = sort_distances(ravel_distances(calculate_distances(np_points)))
    result = calc_min_distance(nodes, distances)

    return result


if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    x = data[1::2]
    y = data[2::2]
    print("{0:.9f}".format(minimum_distance(x, y)))
