import os
import numpy as np
from Var import Var
from sympy import *
from matplotlib import pyplot
from Engine import Engine

if __name__ == '__main__':
    e = Engine(Engine.B, Engine.m, Engine.r)
    var_domain = range(0, len(Var.domain), 10)

    complete_list = Engine.complete_list

    y_listH = np.asarray([e.expression('H', log(e.P), i) for i in var_domain], dtype=float)

    shifted_x_dict, y_dict = dict(), dict()
    atomic_mass_dict, atomic_number_dict, atomic_radii_dict = dict(), dict(), dict()

    for element in complete_list:
        print(element)
        m, z = Engine.library[element].element.mass, Engine.library[element].element.atomic_number
        r = Engine.library[element].atomic_radius
        shifted_x_list = -np.asarray([e.expression(element, log(e.rho) / log(10), i) for i in reversed(var_domain)],
                                     dtype=float) + np.log10(m / z)
        shifted_x_dict[element] = shifted_x_list
        y_dict[element] = np.asarray([e.expression(element, log(e.P) / log(10), i) for i in reversed(var_domain)],
                                     dtype=float)
        atomic_mass_dict[element], atomic_number_dict[element] = m, z
        atomic_radii_dict[element] = r

    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype=float)
    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype=float)
    atomic_radii = np.asarray(list(atomic_radii_dict.values()), dtype=float)

    intersection = (max(arg[0] for arg in shifted_x_dict.values()), min(arg[-1] for arg in shifted_x_dict.values()))
    print(intersection)

    constrained_x_list = np.asarray([x for x in shifted_x_dict['H'] if intersection[0] <= x < intersection[1]],
                                    dtype=float)[:202]
    constrained_y_dict = dict()

    x_list = constrained_x_list

    '''def kernel(x_star, x, denominator):
        return np.exp(-((x - x_star) ** 2) / denominator)


    scale = (x_list[-1] - x_list[0]) / (len(x_list) - 1)
    denominator = 20 * scale ** 2
    kernels = [[kernel(x_list[i], x_list[j], denominator) for j in range(len(x_list))] for i in range(len(x_list))]'''

    for element in complete_list:
        print(element)
        xy_list = list(zip(shifted_x_dict[element], y_dict[element]))
        constrained_y_list = []
        for arg in xy_list:
            if intersection[0] <= arg[0] < intersection[1]:
                constrained_y_list.append(arg[1])
        constrained_y_dict[element] = np.asarray(constrained_y_list[:202], dtype=float)

    normalized_y_dict = {element: constrained_y_dict[element] - constrained_y_dict['H'] for element in complete_list}

    pyplot.subplot(131)
    for element in complete_list:
        pyplot.plot(constrained_x_list, normalized_y_dict[element])

    y_dict = normalized_y_dict
    initial_y_dict = {element: y_dict[element][0] for element in complete_list}
    initial_values = np.asarray(list(initial_y_dict.values()), dtype=float)

    pyplot.subplot(132)
    pyplot.scatter(atomic_numbers, np.exp(initial_values), color='black', s=5)

    pyplot.subplot(133)
    pyplot.scatter(np.log(atomic_numbers), initial_values, color='black', s=5)

    pyplot.show()
