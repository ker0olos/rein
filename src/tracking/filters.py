# forked from https://github.com/jaantollander/OneEuroFilter
import math


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t, x):
        # customizable parameters
        self.beta = 10.0
        self.d_cutoff = 1.0
        self.min_cutoff = 0.004

        # previous values
        self.x_prev = x
        self.dx_prev = 0.0
        self.t_prev = t

    def __call__(self, t, x):
        t_e = t - self.t_prev

        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # the filtered signal
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # update previous values
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
