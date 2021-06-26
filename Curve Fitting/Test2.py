
import os
import numpy as np
import math
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
    # complete_list.remove('Co')
    # complete_list.remove('Xe')
    complete_list = sorted(complete_list, key=lambda arg: Engine.library[arg].element.atomic_number)

    # complete_list = ['D'] + complete_list[20:]

    post_transition_metals = ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At']


    def kernel(x_star, x, scale):
        return np.exp(-((x - x_star) ** 2) / (10 * scale ** 2))

    def smooth(x_values, y_values):
        scale = (x_values[-1] - x_values[0]) / (len(x_values) - 1)
        smooth_y_values = []
        for i in range(len(x_values)):
            kernels = [kernel(x_values[i], x_values[j], scale) for j in range(len(x_values))]
            smooth_y_values.append(sum(kernels[j] * y_values[j] for j in range(len(x_values))) / sum(kernels))
        return smooth_y_values


    x_list = np.asarray([Var.domain[i] for i in range(0, len(Var.domain), 10)], dtype='float')
    dx_list = np.asarray([x_list[i + 1] - x_list[i] for i in range(len(x_list) - 1)], dtype='float')

    y_listH = np.asarray([e.expression('H', log(e.P / (e.z ** (10 / 3))), i) for i in range(0, len(Var.domain), 10)], dtype='float')

    threshold = 2.5
    upper_region = list(filter(lambda arg: arg >= threshold, x_list[:-1]))
    derivative_dict = dict()

    for element in complete_list:
        print(element)
        mass = Engine.library[element].element.mass
        volume_x_list = np.log10(mass) - x_list
        volume_dx_list = [volume_x_list[i + 1] - volume_x_list[i] for i in range(len(volume_x_list) - 1)]
        # y_list = [e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in range(0, len(Var.domain), 10)]
        y_list = [e.expression(element, log(e.P), i) - y_listH[i] for i in range(0, len(Var.domain), 10)]

        derivative_list = [(y_list[i + 1] - y_list[i]) / volume_dx_list[i] for i in range(len(dx_list))]
        # derivative_dict[element] = [derivative_list[i] for i in range(len(derivative_list)) if x_list[i] >= threshold]

        pyplot.subplot(121)
        pyplot.plot(volume_x_list, y_list)

        pyplot.subplot(122)
        # smooth_derivative_list = smooth(volume_x_list[:-1], derivative_list)
        pyplot.plot(volume_x_list[:-1], derivative_list, label=element)

    m = np.vstack([np.asarray(derivative_dict[element], dtype='float') for element in complete_list if element != 'H']).T
    U, S, Vt = np.linalg.svd(m)
    principal_component = -U[:, 0]
    print(f'U: {U[:, 0]}')
    print(f'S: {S}')

    coefficients = []
    coefficient_dict = dict()

    reference_index = float('inf')
    reference_element = ''
    # pyplot.subplot(131)
    for element in complete_list:
        derivative_list = derivative_dict[element]
        ratio = np.asarray([derivative_list[i] / principal_component[i] for i in range(len(derivative_list))], dtype='float')
        coefficients.append((c := np.average(ratio)))
        coefficient_dict[element] = c
        # pyplot.scatter(upper_region, coefficient, s=1, label=element)
        # pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    coefficients = np.asarray(coefficients, dtype='float')
    atomic_numbers = np.asarray([Engine.library[element].element.atomic_number for element in complete_list], dtype='float')
    atomic_masses = np.asarray([Engine.library[element].element.mass for element in complete_list], dtype='float')

    '''pyplot.subplot(142)
    pyplot.scatter(atomic_masses, coefficients)
    pyplot.subplot(143)
    pyplot.scatter(atomic_masses, coefficients ** 2)'''
    # pyplot.subplot(133)
    # pyplot.scatter(np.log(atomic_masses), np.log(coefficients))

    reference_element = 'D'
    print(f'Reference element: {reference_element}')
    Deuterium_coefficient = coefficient_dict[reference_element]
    for element in complete_list:
        coefficient_dict[element] /= Deuterium_coefficient
    coefficients /= Deuterium_coefficient

    y_listRef = np.asarray([e.expression(reference_element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in
                          range(0, len(Var.domain), 10)], dtype='float')
    derivative_listRef = np.asarray([(y_listRef[i + 1] - y_listRef[i]) / dx_list[i] for i in range(len(dx_list))], dtype='float')

    constant_dict = dict()
    for element in complete_list:
        print(element)
        # y_list = np.asarray([e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in
                             # range(0, len(Var.domain), 10)], dtype='float')
        y_list = np.asarray([e.expression(element, log(e.P), i) - y_listH[i] for i in
                             range(0, len(Var.domain), 10)], dtype='float')
        derivative_list = np.asarray([(y_list[i + 1] - y_list[i]) / dx_list[i] for i in range(len(dx_list))], dtype='float')

        linewidth = 3 / (Engine.library[element].element.atomic_number ** 0.5)
        # SECTION: pyplot.subplot(162)
        # pyplot.plot(x_list, y_list - coefficient_dict[element] * y_listRef, linewidth=1)

        # SECTION: pyplot.subplot(163)
        # pyplot.plot(x_list, y_list / coefficient_dict[element], linewidth=linewidth)

        # noinspection PyTypeChecker
        # smooth_derivative_list = smooth(x_list[:-1], derivative_list - coefficient_dict[element] * derivative_listRef)

        # pyplot.subplot(164)
        # pyplot.plot(x_list[:-1], smooth_derivative_list, label=element)
        # pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        residual_list = y_list - coefficient_dict[element] * y_listRef
        constant_dict[element] = np.average(np.asarray([residual_list[i] for i in range(len(residual_list)) if x_list[i] >= threshold]))

    print(coefficient_dict)
    print(constant_dict)
    constants = np.asarray(list(map(constant_dict.__getitem__, complete_list)))

    # SECTION: pyplot.subplot(164)
    # pyplot.scatter(atomic_masses, coefficients, s=3)
    # pyplot.scatter(atomic_masses, constants, s=3)
    # SECTION: pyplot.subplot(165)
    # pyplot.scatter(atomic_masses, constants / coefficients, s=3)

    constantToCoefficientRatio = max(constants / coefficients)
    print(f'Constant to Coefficient Ratio: {constantToCoefficientRatio}')

    y_listRefShifted = y_listRef + constantToCoefficientRatio

    for j in range(len(complete_list)):
        element = complete_list[j]
        print(element)
        mass = Engine.library[element].element.mass

        # y_list = np.asarray([e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in
                             # range(0, len(Var.domain), 10)], dtype='float')
        y_list = np.asarray([e.expression(element, log(e.P), i) - y_listH[i] for i in
                             range(0, len(Var.domain), 10)], dtype='float')

        linewidth = 1 # 5 if element == 'H' else 1 # 3 / (Engine.library[element].element.atomic_number ** 0.5)
        pyplot.subplot(101 + (j // 10) + 10 * math.ceil(len(complete_list) / 10))
        pyplot.plot(np.log10(mass) - x_list, y_list - coefficient_dict[element] * y_listRef - constant_dict[element], linewidth=linewidth, label=element)
        pyplot.ylim(-0.2, 1.2)
        # pyplot.plot(x_list, y_list - coefficient_dict[element] * y_listRefShifted, '--', linewidth=linewidth)

        pyplot.legend(loc='upper right')

    '''for element in complete_list:
        e.plot(element, log(e.P), 131)
        e.plot(element, e.Bp, 132, x_expr=log((1 / e.nP - e.Bp) / e.P))
        e.plot(element, e.Bp / (e.z ** (1 / 3)), 133, x_expr=log((1 / e.nP - e.Bp) / e.P))'''

    pyplot.show()