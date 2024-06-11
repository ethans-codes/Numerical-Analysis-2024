import math
import numpy as np


def monte_carlo_integration(integrand, start_p, end_p, max_iter):
    """Perform Monte Carlo integration of the integrand function between start and end using max_iter samples."""
    # Generate random samples between start_p and end_p
    random_samples = generate_random_numbers(start_p, end_p, max_iter)
    # Evaluate the integrand at each sample
    integrand_values = [integrand(x) for x in random_samples]
    # Calculate the average value of the integrand
    avg = sum(integrand_values) / max_iter
    # Calculate the integral estimate
    integral_estimate = (end_p - start_p) * avg
    # Calculate the error
    standard_error = calculate_error(start_p, end_p, integrand_values, avg, max_iter)

    return integral_estimate, standard_error

def generate_random_numbers(start, end, max_iter):
    """Generate an array of random numbers uniformly distributed between start and end."""
    return np.random.uniform(start, end, max_iter)

def calculate_error(a, b, fx_values, avg, n):
    """Calculate the standard error of the Monte Carlo integration."""
    variance = sum((fx - avg) ** 2 for fx in fx_values) / n
    standard_error = (b - a) * math.sqrt(variance / n)
    return standard_error

def initialize_values():
    # Function to be integrated.
    f = np.sin
    # Starting and ending points
    a = 0
    b = np.pi
    # Maximum number of iterations
    N = 1000000
    return (f, a, b, N)

if __name__ == "__main__":
    """Start the monte carlo"""
    values = initialize_values()
    integral, error = monte_carlo_integration(*values)
    print("The value calculated by Monte Carlo integration is {}.".format(integral))
    print("The error in the Monte Carlo integration is {}.".format(error))
