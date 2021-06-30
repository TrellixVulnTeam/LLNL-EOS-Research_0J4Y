
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

    # complete_list = ['D'] + complete_list[20:]

    post_transition_metals = ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At']

    y_listH = np.asarray([e.expression('H', log(e.P), i) for i in var_domain], dtype='float')

    shifted_x_dict, y_dict = dict(), dict()
    atomic_mass_dict, atomic_number_dict = dict(), dict()

    for element in complete_list:
        print(element)
        m, z = Engine.library[element].element.mass, Engine.library[element].element.atomic_number
        # y_list = [e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in range(0, len(Var.domain), 10)]
        shifted_x_list = -np.asarray([e.expression(element, log(e.rho) / log(10), i) for i in reversed(var_domain)], dtype='float') + np.log10(m / z)
        shifted_x_dict[element] = shifted_x_list
        y_dict[element] = np.asarray([e.expression(element, log(e.P) / log(10), i) for i in reversed(var_domain)], dtype='float')

        atomic_mass_dict[element], atomic_number_dict[element] = m, z

    intersection = (max(arg[0] for arg in shifted_x_dict.values()), min(arg[-1] for arg in shifted_x_dict.values()))
    print(intersection)

    constrained_x_list = np.asarray([x for x in shifted_x_dict['H'] if intersection[0] <= x < intersection[1]], dtype='float')[:202]
    constrained_y_dict = dict()

    pyplot.subplot(121)
    for element in complete_list:
        print(element)
        xy_list = list(zip(shifted_x_dict[element], y_dict[element]))
        constrained_y_list = []
        for arg in xy_list:
            if intersection[0] <= arg[0] < intersection[1]:
                constrained_y_list.append(arg[1])
        constrained_y_dict[element] = np.asarray(constrained_y_list[:202], dtype='float')

        pyplot.plot(constrained_x_list, constrained_y_dict[element])

    normalized_y_dict = {element: constrained_y_dict[element] - constrained_y_dict['H'] for element in complete_list}

    pyplot.subplot(122)
    for element in complete_list:
        pyplot.plot(constrained_x_list, normalized_y_dict[element])

    pyplot.show()
