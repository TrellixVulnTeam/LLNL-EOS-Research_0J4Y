
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

    y_listH = np.asarray([e.expression('H', log(e.P / (e.z ** (10 / 3))), i) for i in range(0, len(Var.domain), 10)],
                         dtype='float')

    threshold = (3, 4)
    threshold_lower_index, threshold_upper_index = 0, 0
    while x_list[threshold_lower_index] < threshold[0]:
        threshold_lower_index += 1
    while x_list[threshold_upper_index] < threshold[1]:
        threshold_upper_index += 1
    upper_region = x_list[threshold_lower_index:threshold_upper_index]

    y_dict = dict()
    y_threshold_dict, derivative_dict = dict(), dict()

    atomic_mass_dict, atomic_number_dict = dict(), dict()

    for element in complete_list:
        print(element)
        # y_list = [e.expression(element, log(e.P / (e.z ** (10 / 3))), i) - y_listH[i] for i in range(0, len(Var.domain), 10)]
        y_dict[element] = (y_list := (np.asarray([e.expression(element, log(e.P), i) for i in range(0, len(Var.domain), 10)],
                            dtype='float') - y_listH))
        derivative_list = np.asarray([y_list[i + 1] - y_list[i] for i in range(len(dx_list))], dtype='float') / dx_list

        y_threshold_dict[element] = y_list[threshold_lower_index:threshold_upper_index]
        derivative_dict[element] = derivative_list[threshold_lower_index:threshold_upper_index]

        pyplot.subplot(161)
        pyplot.plot(x_list, y_list)

        atomic_mass_dict[element] = Engine.library[element].element.mass
        atomic_number_dict[element] = Engine.library[element].element.atomic_number

        # pyplot.subplot(162)

        # smooth_derivative_list = Engine.smooth(x_list[:-1], derivative_list)
        # pyplot.plot(x_list[:-1], smooth_derivative_list, label=element)

    m = np.vstack(
        [np.asarray(derivative_dict[element], dtype='float') for element in complete_list if element != 'H']).T
    U, S, Vt = np.linalg.svd(m)
    principal_component = -U[:, 0]
    print(f'U: {U[:, 0]}')
    print(f'S: {S}')

    coefficient_dict = {element: (derivative_dict[element] @ principal_component) / (np.linalg.norm(principal_component) ** 2) for element in
                        complete_list}

    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype='float')
    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype='float')

    '''pyplot.subplot(162)
    pyplot.scatter(atomic_masses, coefficients)
    pyplot.subplot(163)
    pyplot.scatter(atomic_masses, coefficients ** 2)'''
    # pyplot.subplot(163)
    # pyplot.scatter(np.log(atomic_masses), np.log(coefficients))

    reference_element = 'D'
    print(f'Reference element: {reference_element}')
    Deuterium_coefficient = coefficient_dict[reference_element]
    print(Deuterium_coefficient)
    for element in complete_list:
        coefficient_dict[element] /= Deuterium_coefficient
    coefficients = np.asarray(list(coefficient_dict.values()), dtype='float')

    # pyplot.subplot(162)
    # pyplot.plot(atomic_masses, coefficients)
    # pyplot.subplot(163)
    # pyplot.plot(np.log(atomic_masses), coefficients)

    y_listRef = y_dict['D']
    y_listRefShifted = y_listRef + min(abs(y_listRef))
    derivative_listRef = np.asarray([(y_listRef[i + 1] - y_listRef[i]) / dx_list[i] for i in range(len(dx_list))],
                                    dtype='float')

    constant_dict = dict()
    for element in complete_list:
        print(element)
        y_list = y_dict[element]
        # derivative_list = np.asarray([(y_list[i + 1] - y_list[i]) / dx_list[i] for i in range(len(dx_list))], dtype='float')
        residual_list = y_list - coefficient_dict[element] * y_listRefShifted

        linewidth = 3 / (Engine.library[element].element.atomic_number ** 0.5)

        # pyplot.subplot(162)
        # pyplot.plot(x_list, residual_list, label=element, linewidth=linewidth)

        constant_dict[element] = residual_list[-1]
        # constant_dict[element] = np.average(residual_list[threshold_lower_index:threshold_upper_index])

    print(coefficient_dict)
    print(constant_dict)
    constants = np.asarray(list(map(constant_dict.__getitem__, complete_list)))

    pyplot.subplot(162)
    pyplot.scatter(atomic_masses, coefficients, s=8, color='purple')
    pyplot.subplot(163)
    pyplot.scatter(atomic_masses, constants, s=8, color='black')
    pyplot.subplot(164)
    pyplot.scatter(atomic_masses, constants / coefficients, s=8, color='pink')

    A = np.vstack(((1 / atomic_masses)[3:], np.ones(len(atomic_masses) - 3))).T
    slope, intercept = np.linalg.lstsq(A, (constants / coefficients)[3:], rcond=None)[0]

    pyplot.subplot(165)
    pyplot.scatter(1 / atomic_masses, constants / coefficients, s=8, color='black')
    pyplot.plot([0, 0.5], [intercept, intercept + slope / 2])

    y_listRefShifted2 = y_listRefShifted + intercept
    threshold_y_listRef = y_listRefShifted2[threshold_lower_index:threshold_upper_index]

    selected_list = ['Ta', 'Re', 'Os', 'Pt', 'Au', 'Pb', 'Po']
    s_list_dict = dict()

    # pyplot.scatter(atomic_masses, coefficients, s=3)
    # pyplot.scatter(atomic_masses, constants, s=3)
    # pyplot.scatter(atomic_masses, constants / coefficients, s=3)

    for element in complete_list:
        print(element)
        y_list = (y_dict[element] / coefficient_dict[element] - y_listRefShifted2) * atomic_mass_dict[element]

        pyplot.subplot(166)
        pyplot.plot(x_list, y_list)

    for element in selected_list:
        print(element)
        y_list = (y_dict[element] / coefficient_dict[element] - y_listRefShifted2) * atomic_mass_dict[element]

        s_list_dict[element] = y_list[threshold_lower_index:threshold_upper_index]

        # pyplot.subplot(167)
        # pyplot.plot(x_list, y_list, label=element)
        # pyplot.legend(loc='upper right')

    s_list = np.average(list(s_list_dict.values()), axis=0)
    pyplot.plot(x_list[threshold_lower_index:threshold_upper_index], s_list, color='black')

    coefficient_dict2 = dict()

    for element in complete_list:
        base = threshold_y_listRef + s_list / atomic_mass_dict[element]
        coefficient_dict2[element] = (y_threshold_dict[element] @ base) / (np.linalg.norm(base) ** 2)

    print(coefficient_dict2)

    '''for element in complete_list:
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
        pyplot.subplot(166)
        pyplot.plot(x_list, y_list, linewidth=linewidth)

        # pyplot.subplot(162)
        smooth_derivative_list = Engine.smooth(x_list[:-1], derivative_list)
        c = -1
        for i in range(len(smooth_derivative_list)):
            if abs(smooth_derivative_list[i]) < 0.01:
                c = i
                break
        critical_dict[element] = x_list[c] # x_list[(c := np.argmin(np.abs(smooth_derivative_list)))]'''
    # pyplot.plot(x_list[:-1], smooth_derivative_list, linewidth=linewidth, label=element)

    # pyplot.legend(loc='upper right')

    # pyplot.subplot(161)
    # pyplot.scatter(x_list[c], y_list[c], c='black', marker='o', s=5)

    # pyplot.plot(x_list, y_list - coefficient_dict[element] * y_listRefShifted, '--', linewidth=linewidth)

    '''critical_values = np.asarray(list(critical_dict.values()), dtype='float')
    print(complete_list[20:30])
    # print(list(map(lambda arg: Engine.library[arg].element.ionic_radius, complete_list[20:30])))

    # pyplot.subplot(163)
    # pyplot.plot(atomic_masses[20:30], np.cbrt(atomic_masses / np.power(10, critical_values))[20:30], c='black', marker='o')
    # pyplot.plot(atomic_masses, np.cbrt(atomic_masses / np.power(10, critical_values)), c='black', marker='o')

    # pyplot.subplot(164)
    # pyplot.plot(np.log(atomic_masses), coefficients, marker='o')

    for element in complete_list:
        e.plot(element, log(e.P), 161)
        e.plot(element, e.Bp, 162, x_expr=log((1 / e.nP - e.Bp) / e.P))
        e.plot(element, e.Bp / (e.z ** (1 / 3)), 163, x_expr=log((1 / e.nP - e.Bp) / e.P))'''
    pyplot.show()