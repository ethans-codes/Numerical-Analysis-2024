import math
from colors import bcolors


def trapezoidal_rule(function, point_a, point_b, segment):
    height = (point_b - point_a) / segment  # Calculate height
    sum_of_bases = function(point_a) + function(point_b)  # Summarize start bases
    integral = 0.5 * sum_of_bases  # Initialize with endpoints

    for i in range(1, segment):
        segment_point = point_a + i * height
        integral += function(segment_point)

    integral *= height

    return integral


if __name__ == '__main__':
    f = lambda x: (6 * x ** 2 - math.cos(x ** 4 - x + 2)) / (x ** 2 + x + 2)
    result = trapezoidal_rule(f, -2.8, 1.8, 400)
    print(bcolors.OKBLUE, "Approximate integral:", result, bcolors.ENDC)