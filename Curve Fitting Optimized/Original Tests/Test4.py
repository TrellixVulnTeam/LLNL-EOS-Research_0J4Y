
import os
import numpy as np
from Core.Var import Var
from sympy import *
from matplotlib import pyplot
from Core.Engine import Engine

if __name__ == '__main__':
    e = Engine()

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

    y_dict = dict()
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
        # y_list = [e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in range(0, len(Var.domain), 10)]
        y_list = np.asarray([e.expression(element, log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') - y_listH
        y_dict[element] = y_list

        derivative_list = np.asarray([y_list[i + 1] - y_list[i] for i in range(len(dx_list))], dtype='float') / dx_list
        derivative_dict[element] = derivative_list[threshold_lower_index:threshold_upper_index]

        pyplot.subplot(151)
        pyplot.plot(x_list, y_list)

        atomic_mass_dict[element] = Engine.library[element].element.mass
        atomic_number_dict[element] = Engine.library[element].element.atomic_number

    m = np.vstack([np.asarray(derivative_dict[element], dtype='float') for element in complete_list if element != 'H']).T
    U, S, Vt = np.linalg.svd(m)
    principal_component = -U[:, 0]
    print(f'U: {U[:, 0]}')
    print(f'S: {S}')

    coefficient_dict = {element: np.average(derivative_dict[element] / principal_component) for element in complete_list}

    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype='float')
    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype='float')

    reference_element = 'D'
    print(f'Reference element: {reference_element}')
    Deuterium_coefficient = coefficient_dict[reference_element]
    print(Deuterium_coefficient)
    for element in complete_list:
        coefficient_dict[element] /= Deuterium_coefficient
    coefficients = np.asarray(list(coefficient_dict.values()), dtype='float')

    pyplot.subplot(152)
    pyplot.scatter(atomic_masses, coefficients, s=8, color='purple')

    rgb1 = [163, 160, 255]
    rgb2 = [255, 211, 116]

    density_dict = dict()

    for i in range(len(x_list)):
        y_list = np.asarray([y_dict[element][i] for element in complete_list], dtype=float)

        density_dict[x_list[i]] = y_list

        if i % 8 == 0:
            color1 = '#' + ''.join([hex(256 + int(rgb1[k] * i / len(x_list)))[3:] for k in range(3)])
            color2 = '#' + ''.join([hex(256 + int(rgb2[k] * (1 - i / len(x_list))))[3:] for k in range(3)])

            pyplot.subplot(153)
            pyplot.scatter(atomic_masses, y_list, color=color1, s=5)
            pyplot.plot(atomic_masses, y_list, color=color2, linewidth=0.5)

            divided_y_list = y_list / coefficients
            pyplot.subplot(154)
            pyplot.scatter(atomic_masses, divided_y_list, color=color1, s=5)
            pyplot.plot(atomic_masses, divided_y_list, color=color2, linewidth=0.5)

            if i == 0:
                pyplot.subplot(155)
                pyplot.scatter((1 / atomic_masses)[3:], divided_y_list[3:], color=color1, s=5)
                pyplot.plot((1 / atomic_masses)[3:], divided_y_list[3:], color=color2, linewidth=0.5)

    pyplot.show()
