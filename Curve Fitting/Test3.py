
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
    complete_list = sorted(complete_list, key=lambda arg: Engine.library[arg].element.atomic_number)

    x_list = np.asarray([Var.domain[i] for i in range(0, len(Var.domain), 10)], dtype='float')
    logx_list = np.log(x_list)
    dx_list = np.asarray([x_list[i + 1] - x_list[i] for i in range(len(x_list) - 1)], dtype='float')
    dlogx_list = np.asarray([logx_list[i + 1] - logx_list[i] for i in range(len(logx_list) - 1)], dtype='float')

    y_listH = np.asarray([e.expression('H', log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float')
    # y_listD = np.asarray([e.expression('D', log(e.P), i) for i in range(0, len(Var.domain), 10)],
                         # dtype='float') / y_listH - 1

    y_listD = np.asarray([e.expression('D', log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') / y_listH
    y_listT = np.asarray([e.expression('T', log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') / y_listH

    A = np.vstack((np.log(x_list), np.ones(len(x_list)))).T
    Dm, Db = np.linalg.lstsq(A, y_listD, rcond=None)[0]
    Tm, Tb = np.linalg.lstsq(A, y_listT, rcond=None)[0]

    print(Tm / Dm)

    coefficient_dict = dict()
    for element in complete_list:
        print(element)
        y_list = np.asarray([e.expression(element, log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') / y_listH - 1
        derivative_list = np.asarray([(y_list[i + 1] - y_list[i]) / dlogx_list[i] for i in range(len(dx_list))], dtype='float')
        smooth_derivative_list = Engine.smooth(logx_list[:-1], derivative_list)

        coefficient_dict[element] = smooth_derivative_list[-1]

        pyplot.subplot(141)
        pyplot.plot(x_list, y_list)
        pyplot.subplot(142)
        pyplot.plot(logx_list, y_list)
        pyplot.subplot(143)
        pyplot.plot(logx_list[:-1], smooth_derivative_list)


    Deuterium_coefficient = coefficient_dict['D']
    for element in complete_list:
        coefficient_dict[element] /= Deuterium_coefficient
    print(coefficient_dict)

    y_listRef = np.asarray([e.expression('D', log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') / y_listH - 1

    for element in complete_list:
        print(element)
        y_list = np.asarray([e.expression(element, log(e.P), i) for i in range(0, len(Var.domain), 10)], dtype='float') / y_listH - 1\
            - coefficient_dict[element] * y_listRef
        z = Engine.library[element].element.atomic_number

        pyplot.subplot(144)
        pyplot.plot(x_list, y_list / np.log(z), label=element)
        pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

    pyplot.show()

