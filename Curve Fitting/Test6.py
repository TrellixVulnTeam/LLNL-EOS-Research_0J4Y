
import os
import numpy as np
from Var import Var
from sympy import *
from matplotlib import pyplot
from Engine import Engine

if __name__ == '__main__':
    e = Engine(Engine.B, Engine.m, Engine.r)

    complete_list = []
    for directory in os.listdir('Elements'):
        if len(directory) <= 2:
            complete_list.append(directory)
    complete_list = sorted(complete_list, key=lambda arg: Engine.library[arg].element.atomic_number)
    complete_list[0] = 'H'
    complete_list[1] = 'D'
    complete_list[2] = 'T'

    x_list = np.asarray([Var.domain[i] for i in range(0, len(Var.domain), 10)], dtype='float')
    dx_list = np.asarray([x_list[i + 1] - x_list[i] for i in range(len(x_list) - 1)], dtype='float')

    y_listH = np.asarray([e.expression('H', log(e.P / (e.z ** (10 / 3))), i) for i in range(0, len(Var.domain), 10)], dtype='float')

    threshold = (3, 4)
    threshold_lower_index, threshold_upper_index = 0, 0
    while x_list[threshold_lower_index] < threshold[0]:
        threshold_lower_index += 1
    while x_list[threshold_upper_index] < threshold[1]:
        threshold_upper_index += 1
    upper_region = x_list[threshold_lower_index:threshold_upper_index]
    derivative_dict = dict()

    atomic_mass_dict, atomic_number_dict = dict(), dict()

    for element in complete_list:
        print(element)
        m, z = Engine.library[element].element.mass, Engine.library[element].element.atomic_number
        y_list = np.asarray([e.expression(element, log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float')
        shifted_x_list = np.log10(m / z) - x_list

        pyplot.subplot(121)
        pyplot.plot(shifted_x_list, y_list)

        atomic_mass_dict[element], atomic_number_dict[element] = m, z

    pyplot.subplot(122)
    pyplot.scatter(list(atomic_number_dict.values()), list(atomic_mass_dict.values()), color='black', s=5)

    pyplot.show()
