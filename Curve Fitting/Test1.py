
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

    # complete_list = ['D'] + complete_list[20:]

    post_transition_metals = ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At']

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

    for element in complete_list:
        print(element)
        # y_list = [e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in range(0, len(Var.domain), 10)]
        y_list = np.asarray([e.expression(element, log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') - y_listH
        derivative_list = np.asarray([y_list[i + 1] - y_list[i] for i in range(len(dx_list))], dtype='float') / dx_list
        derivative_dict[element] = derivative_list[threshold_lower_index:threshold_upper_index]

        pyplot.subplot(131)
        pyplot.plot(x_list, y_list)
        # pyplot.subplot(162)

        # smooth_derivative_list = Engine.smooth(x_list[:-1], derivative_list)
        # pyplot.plot(x_list[:-1], smooth_derivative_list, label=element)

    m = np.vstack([np.asarray(derivative_dict[element], dtype='float') for element in complete_list if element != 'H']).T
    U, S, Vt = np.linalg.svd(m)
    principal_component = -U[:, 0]
    print(f'U: {U[:, 0]}')
    print(f'S: {S}')

    coefficient_dict = {element: np.average(derivative_dict[element] / principal_component) for element in complete_list}

    atomic_numbers = np.asarray([Engine.library[element].element.atomic_number for element in complete_list], dtype='float')
    atomic_masses = np.asarray([Engine.library[element].element.mass for element in complete_list], dtype='float')

    '''pyplot.subplot(142)
    pyplot.scatter(atomic_masses, coefficients)
    pyplot.subplot(143)
    pyplot.scatter(atomic_masses, coefficients ** 2)'''
    # pyplot.subplot(143)
    # pyplot.scatter(np.log(atomic_masses), np.log(coefficients))

    reference_element = 'D'
    print(f'Reference element: {reference_element}')
    Deuterium_coefficient = coefficient_dict[reference_element]
    print(Deuterium_coefficient)
    for element in complete_list:
        coefficient_dict[element] /= Deuterium_coefficient
    coefficients = np.asarray(list(coefficient_dict.values()), dtype='float')

    # pyplot.subplot(142)
    # pyplot.plot(atomic_masses, coefficients)
    # pyplot.subplot(143)
    # pyplot.plot(np.log(atomic_masses), coefficients)


    y_listRef = np.asarray([e.expression(reference_element, log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') - y_listH
    y_listRefShifted = y_listRef + min(abs(y_listRef))
    derivative_listRef = np.asarray([(y_listRef[i + 1] - y_listRef[i]) / dx_list[i] for i in range(len(dx_list))], dtype='float')

    constant_dict = dict()
    for element in complete_list:
        print(element)
        # y_list = np.asarray([e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in
                             # range(0, len(Var.domain), 10)], dtype='float')
        y_list = np.asarray([e.expression(element, log(e.P), i) for i in
                             range(0, len(Var.domain), 10)], dtype='float') - y_listH
        # derivative_list = np.asarray([(y_list[i + 1] - y_list[i]) / dx_list[i] for i in range(len(dx_list))], dtype='float')
        residual_list = y_list - coefficient_dict[element] * y_listRefShifted

        linewidth = 3 / (Engine.library[element].element.atomic_number ** 0.5)

        pyplot.subplot(132)
        pyplot.plot(x_list, residual_list, label=element, linewidth=linewidth)

        # SECTION: pyplot.subplot(163)
        # pyplot.plot(x_list, y_list / coefficient_dict[element], linewidth=linewidth)

        # noinspection PyTypeChecker
        # smooth_derivative_list = Engine.smooth(x_list[:-1], derivative_list - coefficient_dict[element] * derivative_listRef)

        # pyplot.subplot(164)
        # pyplot.plot(x_list[:-1], smooth_derivative_list, label=element)
        # pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        constant_dict[element] = np.average(residual_list[threshold_lower_index:threshold_upper_index])

    print(coefficient_dict)
    print(constant_dict)
    constants = np.asarray(list(map(constant_dict.__getitem__, complete_list)))

    # SECTION: pyplot.subplot(164)
    # pyplot.scatter(atomic_masses, coefficients, s=3)
    # pyplot.scatter(atomic_masses, constants, s=3)
    # SECTION: pyplot.subplot(165)
    # pyplot.scatter(atomic_masses, constants / coefficients, s=3)

    critical_dict = dict()

    for element in complete_list:
        print(element)
        mass = Engine.library[element].element.mass

        # y_list = np.asarray([e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in
                             # range(0, len(Var.domain), 10)], dtype='float')
        y_list = np.asarray([e.expression(element, log(e.P), i) for i in
                             range(0, len(Var.domain), 10)], dtype='float')
        y_list = y_list - (y_listH + coefficient_dict[element] * y_listRefShifted + constant_dict[element])
        derivative_list = np.asarray([(y_list[i + 1] - y_list[i]) / dx_list[i] for i in range(len(dx_list))],
                                     dtype='float')

        linewidth = 1 # 5 if element == 'H' else 1 # 3 / (Engine.library[element].element.atomic_number ** 0.5)
        pyplot.subplot(133)
        pyplot.plot(x_list, y_list, linewidth=linewidth)

        # pyplot.subplot(142)
        '''smooth_derivative_list = Engine.smooth(x_list[:-1], derivative_list)
        c = -1
        for i in range(len(smooth_derivative_list)):
            if abs(smooth_derivative_list[i]) < 0.01:
                c = i
                break
        critical_dict[element] = x_list[c] # x_list[(c := np.argmin(np.abs(smooth_derivative_list)))]'''
        # pyplot.plot(x_list[:-1], smooth_derivative_list, linewidth=linewidth, label=element)

        # pyplot.legend(loc='upper right')

        # pyplot.subplot(141)
        # pyplot.scatter(x_list[c], y_list[c], c='black', marker='o', s=5)

        # pyplot.plot(x_list, y_list - coefficient_dict[element] * y_listRefShifted, '--', linewidth=linewidth)

    '''critical_values = np.asarray(list(critical_dict.values()), dtype='float')
    print(complete_list[20:30])
    # print(list(map(lambda arg: Engine.library[arg].element.ionic_radius, complete_list[20:30])))

    # pyplot.subplot(143)
    # pyplot.plot(atomic_masses[20:30], np.cbrt(atomic_masses / np.power(10, critical_values))[20:30], c='black', marker='o')
    # pyplot.plot(atomic_masses, np.cbrt(atomic_masses / np.power(10, critical_values)), c='black', marker='o')

    # pyplot.subplot(144)
    # pyplot.plot(np.log(atomic_masses), coefficients, marker='o')

    for element in complete_list:
        e.plot(element, log(e.P), 141)
        e.plot(element, e.Bp, 142, x_expr=log((1 / e.nP - e.Bp) / e.P))
        e.plot(element, e.Bp / (e.z ** (1 / 3)), 143, x_expr=log((1 / e.nP - e.Bp) / e.P))'''
    pyplot.show()
