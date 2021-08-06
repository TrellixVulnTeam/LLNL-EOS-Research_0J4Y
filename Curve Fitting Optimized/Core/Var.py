
import numpy as np


class Var(object):
    read_domain = [1.3 + i / 1000 for i in range(0, 2801, 5)]
    domain = [1.5 + i / 1000 for i in range(0, 2401, 5)]

    def __init__(self, args):
        self.args = np.log10(args)
        lower, upper = np.searchsorted(self.args[:, 0], [Var.domain[0], Var.domain[-1]], side='right')
        self.args = self.args[lower:upper]
        self.normalized_x_values, self.normalized_y_values = self.args.T

    def __call__(self, normalized_x):
        lower, upper = 0, len(self.args) - 1
        mid = round((lower + upper) / 2)
        while normalized_x != (x_mid := self.normalized_x_values[mid]) and upper - lower > 1:
            if normalized_x > x_mid:
                lower = mid
            else:
                upper = mid
            mid = (lower + upper) // 2
        if normalized_x == x_mid:
            index, ratio = mid, 0
        else:
            index, ratio = lower, (normalized_x - self.normalized_x_values[lower]) / \
                           (self.normalized_x_values[upper] - self.normalized_x_values[lower])
        normalized_y = (1 - ratio) * self.normalized_y_values[index] + ratio * self.normalized_y_values[index + 1]
        return normalized_y

