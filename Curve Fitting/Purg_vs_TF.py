
import math
import numpy as np
from Var import Var
from sympy import *
from matplotlib import pyplot
from matplotlib.colors import rgb2hex
from Engine import Engine

if __name__ == '__main__':
    e = Engine(Engine.B, Engine.m, Engine.r)
    var_domain = range(0, len(Var.domain), 5)

    def color(z):
        k = 2 * np.pi * z / 20
        return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2

    element_list = [element for element in Engine.complete_list if hasattr(Engine.library[element], 'tf_data')][:-8]
    purg_color_dict, tf_color_dict = dict(), dict()

    x_list = np.asarray([Var.domain[i] for i in var_domain], dtype=float)
    reversed_x = np.asarray(list(reversed(x_list)), dtype=float)
    x_list = -reversed_x

    dx = (x_list[-1] - x_list[0]) / (len(x_list) - 1)

    x_dict = dict()
    purg_y_dict, tf_y_dict = dict(), dict()
    purg_dv_dict, tf_dv_dict = dict(), dict()

    atomic_mass_dict, atomic_number_dict = dict(), dict()

    difference_dict = dict()

    for element in element_list:
        print(element)
        library_element = Engine.library[element]

        m, z = library_element.element.mass, library_element.element.atomic_number
        atomic_mass_dict[element], atomic_number_dict[element] = m, z

        purg_color = color(z)
        tf_color = 0.7 * purg_color

        shifted_x = np.log10(m * z) - reversed_x
        purg_y = np.log(np.asarray([library_element.pressure_data[i] for i in reversed(var_domain)], dtype=float)) / np.log(10)
        tf_y = np.log(np.asarray([library_element.tf_data[i] for i in reversed(var_domain)], dtype=float)) / np.log(10)

        purg_derivative = Engine.SMA_smooth(
            np.asarray([purg_y[i + 1] - purg_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx,
            buffer=100)
        tf_derivative = Engine.SMA_smooth(
            np.asarray([tf_y[i + 1] - tf_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx,
            buffer=100)

        x_dict[element], purg_y_dict[element], tf_y_dict[element] = shifted_x, purg_y, tf_y
        purg_dv_dict[element], tf_dv_dict[element] = purg_derivative, tf_derivative

        purg_color_dict[element], tf_color_dict[element] = purg_color, tf_color

        pyplot.subplot(151)
        pyplot.plot(shifted_x, purg_y - (10 / 3) * np.log10(z), color=purg_color)
        pyplot.plot(shifted_x, tf_y - (10 / 3) * np.log10(z), color=tf_color)

        pyplot.subplot(152)
        k = len(x_list) - len(tf_derivative)
        constrained = shifted_x[math.floor(k / 2):-math.ceil(k / 2)]
        pyplot.plot(constrained, purg_derivative, color=purg_color)
        pyplot.plot(constrained, tf_derivative, color=tf_color)

        diff_y = 1 - np.power(10, purg_y - tf_y)
        difference_dict[element] = diff_y
        # diff_derivative = tf_derivative - purg_derivative
        '''diff_derivative = Engine.SMA_smooth(
            np.asarray([diff_y[i + 1] - diff_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx,
            buffer=100)'''

        ax = pyplot.subplot(153)
        # ax.set_ylim(0, 0.16)
        pyplot.plot(shifted_x, diff_y, color=purg_color)

        # pyplot.subplot(154)
        # pyplot.plot(constrained, diff_derivative, color=purg_color)'

    shift_dict = dict()

    print(len(x_list))
    for element in element_list:
        print(element)
        index = -1
        k = -1
        for i in range(len(x_list) - 200):
            x_range = difference_dict['H'][i:i + 200]
            y_range = difference_dict[element][:200]
            A = np.vstack((x_range, np.ones(len(x_range)))).T
            (m, b), residual = np.linalg.lstsq(A, y_range, rcond=None)[:2]
            if (r2 := (1 - residual / (y_range.size * y_range.var()))) > k:
                k = r2
                index = i
        print(k)
        shift = x_list[index] - x_list[0]
        shift_dict[element] = shift

    pyplot.subplot(154)
    for element in element_list:
        pyplot.plot(x_list, difference_dict[element], color=purg_color_dict[element])

    print(shift_dict)
    pyplot.subplot(155)
    pyplot.scatter(np.log10(list(atomic_number_dict.values())), list(shift_dict.values()), color='black', s=3)

    pyplot.show()




