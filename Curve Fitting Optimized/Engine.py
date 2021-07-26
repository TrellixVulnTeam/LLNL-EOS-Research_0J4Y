
import os
import numpy as np
import math
from Library import Library
from Var import Var
from sympy import *
from matplotlib import pyplot


class Engine:
    complete_list = None
    library = Library()

    def __init__(self):
        complete_list = []
        for directory in os.listdir('Purg Data'):
            element = directory[:directory.index('.')]
            if hasattr(Engine.library[element], 'tf_data'):
                complete_list.append(element)
        complete_list = sorted(complete_list, key=lambda arg: Engine.library[arg].element.atomic_number)
        Engine.complete_list = complete_list

    # SECTION: Smoothing ===============================================================================================

    @classmethod
    def kernel_smooth(cls, x_values, y_values, kernels=None):
        if kernels == None:
            def kernel(x_star, x, denominator):
                return np.exp(-((x - x_star) ** 2) / denominator)

            scale = (x_values[-1] - x_values[0]) / (len(x_values) - 1)
            denominator = 20 * scale ** 2
            kernels = [[kernel(x_values[i], x_values[j], denominator) for j in range(len(x_values))] for i in range(len(x_values))]

        buffer = 10
        n = len(y_values)
        smooth_y_values = []

        for i in range(buffer, n - buffer):
            smooth_y_values.append(sum(kernels[i][j] * y_values[j] for j in range(len(y_values))) / sum(kernels[i][:n]))
        smooth_y_values = list(y_values[:buffer]) + smooth_y_values + list(y_values[-buffer:])

        return np.asarray(smooth_y_values, dtype=float)

    @classmethod
    def SMA_smooth(cls, y_values, buffer=80):
        smooth_y_values = []
        total = sum(y_values[:buffer - 1]) + y_values[-1]
        for i in range(-1, len(y_values) - buffer - 1):
            total += (y_values[i + buffer] - y_values[i])
            smooth_y_values.append(total / buffer)
        return np.asarray(smooth_y_values, dtype=float)


