
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
    delta = 0.0001
    units = ['kg', 'm', 's']
    P, B, Bp, rho, h, r, m, z = symbols('P B Bp rho h r m z')
    nP, nB, nBp, nrho, nh, nr, nm, nz = symbols('nP nB nBp nrho nh nr nm nz')
    base_variables = [P, B, Bp, rho, h, r, m, z]
    normalized_variables = {P: nP, B: nB, Bp: nBp, rho: nrho, h: nh, r: nr, m: nm, z: nz}
    var_units = {P  :   {'kg': 1,   'm': -1,    's': -2},
                 B  :   {'kg': 1,   'm': -1,    's': -2},
                 Bp :   {'kg': 0,   'm': 0,     's': 0},
                 rho:   {'kg': 1,   'm': -3,    's': 0},
                 h  :   {'kg': 1,   'm': 2,     's': -1},
                 r  :   {'kg': 0,   'm': 1,     's': 0},
                 m  :   {'kg': 1,   'm': 0,     's': 0},
                 z  :   {'kg': 0,   'm': 0,     's': 0}}

    def __init__(self, *args):
        assert len(args) == len(Engine.units), f'Must have exactly {len(Engine.units)} arguments'
        try:
            self.transform = -np.linalg.inv(np.array([[Engine.var_units[arg][unit] for unit in Engine.units] for arg in args]).T)
        except np.linalg.LinAlgError:
            print('Non-invertible arguments for normalization')
            raise np.linalg.LinAlgError
        self.normalization_variables = args
        self.normalization_dict = dict()
        for variable in Engine.base_variables:
            unit_coefficients = [Engine.var_units[variable][unit] for unit in Engine.units]
            normalization_coefficients = np.matmul(self.transform, unit_coefficients)
            c = gcd([nsimplify(coeff) for coeff in normalization_coefficients])
            if c != 0:
                self.normalization_dict[variable] = [round(1 / c)] + [round(coeff / c) for coeff in normalization_coefficients]
            else:
                self.normalization_dict[variable] = [1] + [0] * len(self.normalization_variables)
        self.var_dict = dict()
        for base_var in Engine.base_variables:
            self.var_dict[base_var] = lambda element, index, var=base_var: Engine.base(element, var, index)
            normalized_var = Engine.normalized_variables[base_var]
            self.var_dict[normalized_var] = lambda element, index, var=base_var: self.normalized(element, var, index)

        complete_list = []
        for directory in os.listdir('Elements'):
            if len(directory) <= 2:
                complete_list.append(directory)
        complete_list = sorted(complete_list, key=lambda arg: Engine.library[arg].element.atomic_number)
        complete_list[0] = 'H'
        complete_list[1] = 'D'
        complete_list[2] = 'T'
        Engine.complete_list = complete_list

    # SECTION: Smoothing ===============================================================================================

    @classmethod
    def smooth(cls, x_values, y_values, kernels=None):
        if kernels == None:
            def kernel(x_star, x, denominator):
                return np.exp(-((x - x_star) ** 2) / denominator)

            scale = (x_values[-1] - x_values[0]) / (len(x_values) - 1)
            denominator = 20 * scale ** 2
            kernels = [[kernel(x_values[i], x_values[j], denominator) for j in range(len(x_values))] for i in range(len(x_values))]

        smooth_y_values = []
        for i in range(len(x_values)):
            smooth_y_values.append(sum(kernels[i][j] * y_values[j] for j in range(len(x_values))) / sum(kernels[i]))
        return smooth_y_values

    # SECTION: Base functions ==========================================================================================

    @classmethod
    def base(cls, element, variable, index):
        try:
            Engine.library[element][variable][index]
        except IndexError:
            print(index)
        return Engine.library[element][variable][index]

    # SECTION: Normalized functions ====================================================================================

    def normalized(self, element, variable, index):
        element = Engine.library[element]
        normalization_factor = np.product([element[self.normalization_variables[i]][index] **
                                           self.normalization_dict[variable][i + 1] for i in range(len(self.normalization_variables))])
        return (element[variable][index] ** self.normalization_dict[variable][0]) * normalization_factor

    # SECTION: Generic Expressions =====================================================================================

    def expression(self, element, expr, index):
        return expr.subs([(variable, self.var_dict[variable](element, index)) for variable in expr.free_symbols])

    # SECTION: Plotting methods ========================================================================================

    def plot(self, element, y_expr, n, x_expr=None, scatter=False, density_range=(1.5, 3)):
        if not x_expr:
            x_expr = log(self.rho) / log(10)
        x_min, x_max = max(density_range[0], Var.domain[0]), min(density_range[1], Var.domain[-1])
        i_min = math.floor((len(Var.domain) - 1) * (x_min - Var.domain[0]) / (Var.domain[-1] - Var.domain[0]))
        i_max = math.ceil((len(Var.domain) - 1) * (x_max - Var.domain[0]) / (Var.domain[-1] - Var.domain[0]))
        x_list = [self.expression(element, x_expr, i) for i in range(i_min, i_max, 10)]
        y_list = [self.expression(element, y_expr, i) for i in range(i_min, i_max, 10)]

        pyplot.subplot(n)
        subplot = pyplot.gcf().get_axes()[n % 10 - 1]
        if not subplot.get_title():
            subplot.set_title(f'{element}: {y_expr} vs. {x_expr}')
        elif str(y_expr) not in subplot.get_title():
            subplot.set_title(f'{subplot.get_title()} and {y_expr}')
        else:
            subplot.set_title(f'{y_expr} vs {x_expr}')
        subplot.set_xlabel(str(x_expr))
        subplot.set_ylabel(str(y_expr))

        if scatter:
            pyplot.scatter(x_list, y_list, s=1)
        else:
            pyplot.plot(x_list, y_list)

    def derivative_plot(self, element, y_expr, n, x_expr=None, scatter=False, density_range=(1.5, 3)):
        if not x_expr:
            x_expr = self.rho
        i_min = math.floor((len(Var.domain) - 1) * (density_range[0] - Var.domain[0]) / (Var.domain[-1] - Var.domain[0]))
        i_max = math.ceil((len(Var.domain) - 1) * (density_range[1] - Var.domain[0]) / (Var.domain[-1] - Var.domain[0]))
        x_list = [self.expression(element, x_expr, i) for i in range(i_min, i_max, 20)]
        y_list = []
        for i in range(i_min, i_max, 20):
            try:
                dy = self.expression(element, y_expr, i + 50) - self.expression(element, y_expr, i - 50)
                dx = self.expression(element, x_expr, i + 50) - self.expression(element, x_expr, i - 50)
                y_list.append(dy / dx)
            except IndexError:
                y_list.append(0)

        pyplot.subplot(n)
        subplot = pyplot.gcf().get_axes()[n % 10 - 1]
        if not subplot.get_title():
            subplot.set_title(f'{element}: d({y_expr}) / d{x_expr}')
        elif str(y_expr) not in subplot.get_title():
            subplot.set_title(f'{subplot.get_title()} and d({y_expr}) / d({x_expr})')
        else:
            subplot.set_title(f'd({y_expr}) / d({x_expr})')
        subplot.set_xlabel(str(x_expr))
        subplot.set_ylabel(f'd({y_expr}) / d({x_expr})')
        subplot.set_ylim([0, 0.05])

        if scatter:
            pyplot.scatter(x_list, y_list, s=3)
        else:
            pyplot.plot(x_list, y_list)


