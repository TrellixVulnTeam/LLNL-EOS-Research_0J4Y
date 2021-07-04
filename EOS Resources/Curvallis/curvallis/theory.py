
import math
from sympy import *


class Processor(object):
    _var_x = symbols('x')
    _estimation_coefficient = 0.002

    def __init__(self, args, func=lambda x: Integer(0)):
        self._args = sorted(args, key=lambda arg: arg[0])
        self._xList = [arg[0] for arg in self._args]
        self._yList = [arg[1] for arg in self._args]
        self._length = len(args)
        self._range = (self._xList[0], self._xList[self._length - 1])
        self._approximation = func(self._var_x)

    def _x_binary_search(self, x):
        """
        Search for indices containing x -- O(log N)
        """
        lower, upper = 0, self._length - 1
        index = math.floor((lower + upper) / 2)
        while ((mid := self._xList[index]) != x and upper - lower != 1):
            if x < mid:
                upper = index
            else:
                lower = index
            index = math.floor((lower + upper) / 2)
        return (index, index + 1) if mid == x else (lower, upper)

    def function(self, x):
        """
        ind = self._x_binary_search(x)
        try:
            k = (x - self._xList[ind[0]]) / (self._xList[ind[1]] - self._xList[ind[0]])
            return k * self._yList[ind[1]] + (1 - k) * self._yList[ind[0]]
        except IndexError:
            return self._yList[ind[0]]
        """
        return float(self._approximation.subs(self._var_x, x))

    def first_derivative(self, x):
        """
        ind = self._x_binary_search(x)
        if ind[1] == self._length:
            ind = (self._length - 2, self._length - 1)
        return (self._yList[ind[1]] - self._yList[ind[0]]) / (self._xList[ind[1]] - self._xList[ind[0]])
        """
        return float(diff(self._approximation).subs(self._var_x, x))

    def logarithmic_derivative(self, x):
        """
        return self.first_derivative(x) / self.function(x)
        """
        return float((diff(self._approximation) / self._approximation).subs(self._var_x, x))

    def estimation_range(self):
        df_f = (diff(self._approximation) / self._approximation).simplify()
        log_curvature = diff(df_f, self._var_x, 2) / ((1 + diff(df_f) ** 2) ** (3 / 2))

        def curvature(x):
            return log_curvature.subs(self._var_x, x)
        index = self._length - 1
        while curvature(self._xList[index]) < self._estimation_coefficient * 10:
            index -= 1
        target_curvature = (self._estimation_coefficient * curvature(self._range[1])) ** 0.5
        lower, upper = self._xList[index], self._range[1]
        mid = (lower + upper) / 2
        while abs((c := curvature(mid)) - target_curvature) > target_curvature / 100:
            if c > target_curvature:
                lower = mid
            else:
                upper = mid
            mid = (lower + upper) / 2
        return (mid, self._range[1])

if __name__ == '__main__':
    n = 20
    func = lambda x: x ** 2
    squares = [(i, func(i)) for i in range(n + 1)]
    p = Processor(squares, func)
    print(p.estimation_range())



