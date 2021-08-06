
import mendeleev
import numpy as np
import math
from matplotlib import pyplot
from Core.Engine import Engine

if __name__ == '__main__':
    element_list = ['H', 'Fe', 'Sn', 'Ta', 'Pb']
    data = {element: np.loadtxt(f'TF Data/{element}_TFCC_rhoPccgs.dat') for element in element_list}
    x_list = np.asarray([data['H'][i][0] for i in range(0, len(data['H']), 10)], dtype=float)
    x_list = np.log10(x_list)
    reversed_x_list = np.asarray(list(reversed(x_list)), dtype=float)
    y_dict = {element: np.asarray([data[element][i][1] for i in range(0, len(data[element]), 10)], dtype=float) for element in element_list}

    atomic_mass_dict = {element: getattr(mendeleev, element).mass for element in element_list}
    atomic_number_dict = {element: getattr(mendeleev, element).atomic_number for element in element_list}

    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype=float)
    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype=float)

    x_list = -reversed_x_list
    dx_list = np.asarray([x_list[i + 1] - x_list[i] for i in range(len(x_list) - 1)], dtype=float)

    buffer = 20
    def kernel(x_star, x, denominator):
        return np.exp(-((x - x_star) ** 2) / denominator)

    scale = (x_list[-1] - x_list[0]) / (len(x_list) - 1)
    denominator = 200 * scale ** 2
    kernels = [[kernel(x_list[i], x_list[j], denominator) for j in range(len(x_list))] for i in range(len(x_list))]

    reversed_y_dict = {element: np.asarray(list(reversed(np.log10(y_dict[element]))), dtype=float) for element in element_list}
    '''reversed_y_dict = {
        element: np.asarray(Engine.smooth(x_list, reversed_y_dict[element], kernels=kernels), dtype=float) for
        element in element_list}'''

    derivative_dict = dict()
    second_derivative_dict = dict()

    critical_point_dict = dict()

    for element in element_list:
        y_list = reversed_y_dict[element]
        derivative_list = [(y_list[i + 1] - y_list[i]) / dx_list[i] for i in range(len(y_list) - 1)]
        derivative_list = np.asarray(Engine.SMA_smooth(derivative_list), dtype=float)

        second_derivative_list = [(derivative_list[i + 1] - derivative_list[i]) / dx_list[i] for i in
                                  range(len(derivative_list) - 1)]
        second_derivative_list = np.asarray(Engine.SMA_smooth(second_derivative_list), dtype=float)

        derivative_dict[element], second_derivative_dict[element] = derivative_list, second_derivative_list

        k = math.floor((len(x_list) - len(second_derivative_list)) / 2)
        critical_point_dict[element] = x_list[k + np.argmin(second_derivative_list)]

        m, z = atomic_mass_dict[element], atomic_number_dict[element]

        pyplot.subplot(161)
        pyplot.plot(x_list, y_list)

        pyplot.subplot(162)
        k = len(x_list) - len(derivative_list)
        pyplot.plot(x_list[math.floor(k / 2):-math.ceil(k / 2)], derivative_list)

        pyplot.subplot(163)
        k = len(x_list) - len(second_derivative_list)
        pyplot.plot(x_list[math.floor(k / 2):-math.ceil(k / 2)], second_derivative_list)

        pyplot.subplot(164)
        pyplot.plot(x_list[math.floor(k / 2):-math.ceil(k / 2)] + np.log10(m), second_derivative_list)

    print(critical_point_dict)
    critical_values = np.asarray(list(critical_point_dict.values()), dtype=float)
    shifted_critical_values = critical_values + np.log10(atomic_masses)

    pyplot.subplot(165)
    pyplot.plot(np.log10(atomic_numbers)[1:], shifted_critical_values[1:], color='black', marker='o', markersize=3)

    for element in element_list:
        y_list = reversed_y_dict[element]

        m, z = atomic_mass_dict[element], atomic_number_dict[element]

        pyplot.subplot(166)
        k = len(x_list) - len(y_list)
        if k != 0:
            pyplot.plot(x_list[math.floor(k / 2):-math.ceil(k / 2)] + np.log10(m * z), y_list)
        else:
            pyplot.plot(x_list + np.log10(m * z), y_list)

    pyplot.show()

