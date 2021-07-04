import os
import numpy as np
from Var import Var
from sympy import *
from matplotlib import pyplot
from Engine import Engine

if __name__ == '__main__':
    e = Engine(Engine.B, Engine.m, Engine.r)
    var_domain = range(0, len(Var.domain), 10)

    complete_list = []
    for directory in os.listdir('Elements'):
        if len(directory) <= 2:
            complete_list.append(directory)
    complete_list = sorted(complete_list, key=lambda arg: Engine.library[arg].element.atomic_number)
    complete_list[0] = 'H'
    complete_list[1] = 'D'
    complete_list[2] = 'T'

    # complete_list = ['D'] + complete_list[20:]

    post_transition_metals = ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At']

    y_listH = np.asarray([e.expression('H', log(e.P), i) for i in var_domain], dtype=float)

    shifted_x_dict, y_dict = dict(), dict()
    atomic_mass_dict, atomic_number_dict, atomic_radii_dict = dict(), dict(), dict()

    for element in complete_list:
        print(element)
        m, z = Engine.library[element].element.mass, Engine.library[element].element.atomic_number
        r = Engine.library[element].atomic_radius
        # y_list = [e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in range(0, len(Var.domain), 10)]
        shifted_x_list = -np.asarray([e.expression(element, log(e.rho) / log(10), i) for i in reversed(var_domain)],
                                     dtype=float) + np.log10(m / z)
        shifted_x_dict[element] = shifted_x_list
        y_dict[element] = np.asarray([e.expression(element, log(e.P) / log(10), i) for i in reversed(var_domain)],
                                     dtype=float)
        atomic_mass_dict[element], atomic_number_dict[element] = m, z
        atomic_radii_dict[element] = r

    intersection = (max(arg[0] for arg in shifted_x_dict.values()), min(arg[-1] for arg in shifted_x_dict.values()))
    print(intersection)

    constrained_x_list = np.asarray([x for x in shifted_x_dict['H'] if intersection[0] <= x < intersection[1]],
                                    dtype=float)[:202]
    constrained_y_dict = dict()

    x_list = constrained_x_list


    def kernel(x_star, x, denominator):
        return np.exp(-((x - x_star) ** 2) / denominator)


    scale = (x_list[-1] - x_list[0]) / (len(x_list) - 1)
    denominator = 20 * scale ** 2
    kernels = [[kernel(x_list[i], x_list[j], denominator) for j in range(len(x_list))] for i in range(len(x_list))]

    for element in complete_list:
        print(element)
        xy_list = list(zip(shifted_x_dict[element], y_dict[element]))
        constrained_y_list = []
        for arg in xy_list:
            if intersection[0] <= arg[0] < intersection[1]:
                constrained_y_list.append(arg[1])
        constrained_y_dict[element] = np.asarray(Engine.smooth(x_list, constrained_y_list[:202], kernels=kernels),
                                                 dtype=float)

    normalized_y_dict = {element: constrained_y_dict[element] - constrained_y_dict['H'] for element in complete_list}

    pyplot.subplot(141)
    for element in complete_list:
        pyplot.plot(constrained_x_list, normalized_y_dict[element])

    y_dict = normalized_y_dict

    x_list = constrained_x_list
    dx_list = np.asarray([x_list[i + 1] - x_list[i] for i in range(len(x_list) - 1)], dtype=float)

    threshold = (intersection[0], -3)
    threshold_lower_index, threshold_upper_index = 0, 0
    while x_list[threshold_lower_index] < threshold[0]:
        threshold_lower_index += 1
    while x_list[threshold_upper_index] < threshold[1]:
        threshold_upper_index += 1

    shifted_y_dict = dict()

    for element in complete_list:
        print(element)
        # dy_list = np.asarray([y_list[i + 1] - y_list[i] for i in range(len(y_list) - 1)], dtype=float)
        # derivative_list = dy_list / dx_list
        shifted_y_list = y_dict[element] - y_dict[element][0]
        shifted_y_dict[element] = shifted_y_list

        # pyplot.plot(x_list, shifted_y_list)

    m = np.vstack(list(shifted_y_dict.values())).T
    U, S, Vt = np.linalg.svd(m)
    principal_component = U[:, 0]
    print(f'U: {U[:, 0]}')
    print(f'S: {S}')

    coefficient_dict = {element: shifted_y_dict[element] @ principal_component / (np.linalg.norm(principal_component) ** 2)
                        for element in complete_list}

    coefficients = np.asarray(list(coefficient_dict.values()), dtype=float)

    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype=float)
    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype=float)
    atomic_radii = np.asarray(list(atomic_radii_dict.values()), dtype=float)

    rgb1 = [163, 160, 255]
    rgb2 = [255, 211, 116]

    density_dict = dict()

    pyplot.subplot(142)
    for i in range(len(x_list)):
        y_list = np.asarray([y_dict[element][i] - y_dict[element][0] for element in complete_list], dtype=float)

        density_dict[x_list[i]] = y_list

        if i % 8 == 0:
            color1 = '#' + ''.join([hex(256 + int(rgb1[k] * i / len(x_list)))[3:] for k in range(3)])
            color2 = '#' + ''.join([hex(256 + int(rgb2[k] * (1 - i / len(x_list))))[3:] for k in range(3)])

            pyplot.scatter(atomic_numbers, y_list, color=color1, s=5)
            pyplot.plot(atomic_numbers, y_list, color=color2, linewidth=0.5)

    '''m = np.vstack(list(density_dict.values())).T
    U, S, Vt = np.linalg.svd(m)
    density_principal_component = -U[:, 0]
    print(f'U: {density_principal_component}')
    print(f'S: {S}')

    c = np.linalg.norm(density_principal_component) ** 2
    density_coefficient_dict = {k: density_dict[k] @ density_principal_component / c for k in density_dict.keys()}

    principal_component = np.asarray(list(density_coefficient_dict.values()), dtype=float)

    c = np.linalg.norm(principal_component) ** 2
    coefficient_dict = {element: shifted_y_dict[element] @ principal_component / c for element in complete_list}
    coefficients = np.asarray(list(coefficient_dict.values()), dtype=float)
    print(coefficient_dict)'''

    pyplot.subplot(143)
    pyplot.scatter(atomic_numbers, list(coefficient_dict.values()), color='black', s=5)

    for i in range(len(x_list)):
        y_list = np.asarray([y_dict[element][i] - y_dict[element][0] for element in complete_list], dtype=float) / coefficients
        if i % 8 == 0:
            color1 = '#' + ''.join([hex(256 + int(rgb1[k] * i / len(x_list)))[3:] for k in range(3)])
            color2 = '#' + ''.join([hex(256 + int(rgb2[k] * (1 - i / len(x_list))))[3:] for k in range(3)])

            pyplot.subplot(144)
            pyplot.scatter(atomic_numbers[3:], y_list[3:], color=color1, s=5)
            pyplot.plot(atomic_numbers[3:], y_list[3:], color=color2, linewidth=0.5)

    pyplot.plot(atomic_numbers, atomic_radii / 10000, color='black', linewidth=0.5, marker='o', markersize=3)

    pyplot.show()
