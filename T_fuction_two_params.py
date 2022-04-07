import numpy.polynomial.chebyshev as cheb
import collections

# T fuction for two params, for two params (A, B) with importance (a, b), with a + b == 1 and a>= b,
# it takes b and returns time 0 <= t <= 1 of the first phase of round - optimization A and B (second phase of round is optimizing A alone)
T_interpolated = cheb.Chebyshev([ 2.75252996e-01,  1.33025930e-01,  1.57681534e-02, -5.02264965e-03,
        1.70947758e-03, -7.18750165e-04,  3.58357948e-04, -1.74951220e-04,
        8.31216330e-05, -3.78504078e-05,  1.45678360e-05, -4.26413119e-06,
        5.48183233e-07, -3.83188198e-07,  2.17116350e-06, -4.52963128e-06,
        6.23581649e-06, -6.45094202e-06,  5.23893664e-06, -3.30015201e-06], domain=[0. , 0.5], window=[-1,  1])


def time_form_importance_two_params(importance_values):
    assert(len(importance_values) == 2)
    a, b = importance_values
    assert(a >= b)
    bias = 0.05
    t_b = T_interpolated(max(b, bias))
    return 1 - t_b, t_b


# takes as argument importance dict and returns time dict
def make_dist_two_params(importance):
    assert(len(importance) == 2)
    keys = list(importance.keys())
    values = list(importance.values())
    time_values = time_form_importance_two_params(values)
    return collections.OrderedDict(zip(keys, time_values))


