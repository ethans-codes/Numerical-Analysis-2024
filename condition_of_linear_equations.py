import numpy as np
from inverse_matrix import inverse
from colors import bcolors
from matrix_utility import MaxNorm

def condition_number(A):
    # Step 1: Calculate the max norm (infinity norm) of A
    norm_A = MaxNorm(A)

    # Step 2: Calculate the inverse of A
    A_inv = inverse(A)

    # Step 3: Calculate the max norm of the inverse of A
    norm_A_inv = MaxNorm(A_inv)

    # Step 4: Compute the condition number
    cond = norm_A * norm_A_inv

    print(bcolors.YELLOW,"=====================================================================================================================")
    print("\nOriginal matrix A: \n",bcolors.ENDC, A)

    print(bcolors.OKBLUE,"=====================================================================================================================")
    print("\nInverse of matrix A: \n",bcolors.ENDC, A_inv)

    print(bcolors.OKGREEN,"------------------------------------------------------------------------------------------------------------------",bcolors.ENDC)
    print(bcolors.HEADER, "Norm of A:", bcolors.ENDC, norm_A)
    print(bcolors.HEADER, "Norm of the inverse of A:", bcolors.ENDC, norm_A_inv)
    return cond

if __name__ == '__main__':
    A = np.array([[2, 1.7, -2.5],
                  [1.24, -2, -0.5],
                  [3, 0.2, 1]])
    cond = condition_number(A)
    print(bcolors.HEADER, "Condition number: ",bcolors.ENDC, cond)