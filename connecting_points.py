# Uses python3
import sys
import numpy as np
import pandas as pd


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
    excel = pd.ExcelWriter('data1.xlsx')
    path = []
    all_data = []
    data = pd.DataFrame(0, index=range(len(nodes)), columns=range(len(nodes)))
    first_node = nodes[int(distances[0][0])]
    used = set([first_node])
    first_node.location = first_node
    total_distance = 0
    i = 0
    while len(used) < len(nodes):
        froms, tos, dist = distances[i]
        froms = int(froms)
        tos = int(tos)
        row_of_all_data = [froms, tos, dist,
                           nodes[froms].location, nodes[tos].location]
        if is_valid_connection(nodes[tos]):
            if froms < tos:
                path.append([froms, tos, dist])
                data.iloc[froms, tos] = dist
            else:
                path.append([tos, froms, dist])
                data.iloc[tos, froms] = dist
            nodes[tos].location = nodes[froms]
            used.add(nodes[tos])
            row_of_all_data.extend([
                nodes[froms].location, nodes[tos].location, True])
            total_distance += dist
        elif is_valid_connection(nodes[froms]):
            if froms < tos:
                path.append([froms, tos, dist])
                data.iloc[froms, tos] = dist
            else:
                path.append([tos, froms, dist])
                data.iloc[tos, froms] = dist
            nodes[froms].location = nodes[tos]
            used.add(nodes[froms])
            total_distance += dist
            row_of_all_data.extend([
                nodes[froms].location, nodes[tos].location, True])
        else:
            row_of_all_data.extend([
                nodes[froms].location, nodes[tos].location, False])

        i += 1
        all_data.append(row_of_all_data)
    path = pd.DataFrame(path)
    col_names = ['from', 'to', 'dist', 'start from.location',
                 'start to.location', 'end from.location', 'end to.location', 'used']
    all_data = pd.DataFrame(all_data, columns=col_names)
    data.to_excel(excel, sheet_name='my path graph')
    path.to_excel(excel, sheet_name='my path')
    all_data.to_excel(excel, sheet_name='all data')
    excel.save()
    excel.close()
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
