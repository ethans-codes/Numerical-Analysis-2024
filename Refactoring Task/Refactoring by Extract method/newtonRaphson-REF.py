from colors import bcolors


def newton_raphson(x, f, df, start, end, eps=1e-6, max_iter=50):
    print("{:<10} {:<15} {:<15} ".format("Iteration", "xo", "x1"))

    for i in range(max_iter):
        if df(x) == 0:
            if f(x) == 0:
                return round(x, 6)  # Procedure completed successfully

            else:
                print("Derivative is zero at x0, method cannot continue.")
                return

        x1 = x - f(x) / df(x)

        if out_of_range(x1, start, end):
            return None  # Out of range can't find roots

        elif check_found(x, x1, eps):
            return round(x1, 6)  # Procedure completed successfully

        else:
            # Continue to the next iteration
            print("{:<10} {:<15.9f} {:<15.9f} ".format(i, x, x1))
            x = x1

    return round(x1, 6)


def out_of_range(x1, start, end):
    if end < x1 < start:
        return True
    else:
        return False


def check_found(x0, x1, eps):
    if abs(x1 - x0) < eps:
        return True
    else:
        return False


def Start_Newton_Raphson():
    args = get_arguments()  # Receiving arguments
    roots = []  # Initializing empty list for roots

    # args[2], args[3] - starting and ending points
    for x in range(args[2], args[3] + 1):
        # for each x in range of stating and ending points find root
        root = newton_raphson(x, *args)
        # check if found root is valid
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def get_arguments():
    """
    In ideal the function supposed to receive the input data from the user
    However the implication is different and not part of refactoring
    """
    f = lambda x: x ** 2 - 5 * x + 2  # the function to calculate
    df = lambda x: 2 * x - 5          # the derivative of the function
    start = 0                         # starting point
    end = 5                           # ending point
    eps = 1e-6                        # tolerance
    max_iter = 100                    # maximum number of iterations
    return (f, df, start, end, eps, max_iter)


if __name__ == '__main__':
    try:
        result = Start_Newton_Raphson()
        if result:
            print(bcolors.OKBLUE + "\nThe equation f(x) has approximate roots at x = {}".format(result) + bcolors.ENDC)
        else:
            print("No roots found in the specifieSd range.")
    except:
        print("Error occurred during the calculation.")
