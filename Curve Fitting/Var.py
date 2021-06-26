
import numpy as np


class Var(object):
    domain = [1.5 + i / 1000 for i in range(2501)]

    def __init__(self, args):
        self.args = sorted(args, key=lambda arg: arg[0])
        self.args = list(filter(lambda arg: Var.domain[0] - 0.1 <= np.log10(arg[0]) <= Var.domain[-1] + 0.1, self.args))
        self.x_values = [arg[0] for arg in self.args]
        self.y_values = [arg[1] for arg in self.args]

        self.normalized_x_values = np.log10(self.x_values)

        if (r := self.y_values[-1] / self.y_values[0]) > 0 and abs(np.log10(r)) > 1:
            self.normalized_y_values = np.log10(self.y_values)
            self.y_transform = np.log10
            self.y_inverse_transform = lambda y: np.power(10, y)
        else:
            self.normalized_y_values = self.y_values
            self.y_transform = lambda y: y
            self.y_inverse_transform = lambda y: y

    def __call__(self, x):
        normalized_x = np.log10(x)
        lower, upper = 0, len(self.x_values) - 1
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
        return self.y_inverse_transform(normalized_y)

