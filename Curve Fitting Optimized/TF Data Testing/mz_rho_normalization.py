
import math
import numpy as np
from Core.Var import Var
from sympy import *
from matplotlib import pyplot
from Core.Engine import Engine

if __name__ == '__main__':
    e = Engine(Engine.B, Engine.m, Engine.r)
    var_domain = range(0, len(Var.domain), 10)

    # complete_list = Engine.complete_list
    complete_list = Engine.complete_list[:8]

    x_list = np.asarray([Var.domain[i] for i in var_domain], dtype=float)
    reversed_x = np.asarray(list(reversed(x_list)), dtype=float)

    dx = (x_list[-1] - x_list[0]) / (len(x_list) - 1)

    shifted_x_dict, shifted_y_dict = dict(), dict()
    derivative_dict, second_derivative_dict = dict(), dict()

    atomic_mass_dict, atomic_number_dict = dict(), dict()

    for element in complete_list:
        print(element)
        m, z = Engine.library[element].element.mass, Engine.library[element].element.atomic_number

        shifted_x = np.log10(m * z) - reversed_x
        shifted_y = np.asarray([e.expression(element, log(e.P), i) for i in reversed(var_domain)], dtype=float) / np.log(10)
        shifted_x_dict[element], shifted_y_dict[element] = shifted_x, shifted_y

        derivative = np.asarray([shifted_y[i + 1] - shifted_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx
        derivative = Engine.SMA_smooth(derivative, buffer=100)

        second_derivative = np.asarray([derivative[i + 1] - derivative[i] for i in range(len(derivative) - 1)], dtype=float) / dx
        second_derivative = Engine.SMA_smooth(second_derivative, buffer=100)

        derivative_dict[element], second_derivative_dict[element] = derivative, second_derivative

        atomic_mass_dict[element], atomic_number_dict[element] = m, z

        pyplot.subplot(141)
        pyplot.plot(-reversed_x, shifted_y)

        pyplot.subplot(142)
        k = len(x_list) - len(derivative)
        pyplot.plot(-reversed_x[math.floor(k / 2):-math.ceil(k / 2)], derivative)

        pyplot.subplot(143)
        k = len(x_list) - len(second_derivative)
        pyplot.plot(-reversed_x[math.floor(k / 2):-math.ceil(k / 2)], second_derivative)

        pyplot.subplot(144)
        k = len(x_list) - len(derivative)
        pyplot.plot(shifted_x[math.floor(k / 2):-math.ceil(k / 2)], derivative)

    pyplot.show()



