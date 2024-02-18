from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, swap_rows_elementary_matrix, is_matrix_square, MultiplyMatrix
import numpy as np

e_matrices = []
e_count = 0


def inverse(matrix):
    print(bcolors.OKBLUE,
          f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n",
          bcolors.ENDC)
    if not (is_matrix_square(matrix)):
        raise ValueError(bcolors.FAIL, "Input matrix isn't square or empty", bcolors.ENDC)

    global e_matrices, e_count
    n = len(matrix)
    inverse_mat = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    # direction means first above and then below diagonal
    for direction in [1, -1]:
        start, end, step = (0, n, 1) if direction == 1 else (n - 1, -1, -1)
        for i in range(start, end, step):
            # Check if the diagonal element is zero, if yes, swap rows
            if matrix[i, i] == 0:
                # Find a row below with a non-zero element in the same column and swap them
                for k in range(i + 1, n):
                    if matrix[k, i] != 0:
                        e_matrices.append(swap_rows_elementary_matrix(n, i, k))
                        e_count += 1
                        print(
                            f"elementary matrix E{e_count} to swap R{i + 1} and R{k + 1}:\n {e_matrices[e_count - 1]} \n")
                        matrix = MultiplyMatrix(e_matrices[e_count - 1], matrix)
                        inverse_mat = MultiplyMatrix(e_matrices[e_count - 1], inverse_mat)
                        print(f"The matrix after elementary operation :\n {matrix}")
                        print(bcolors.OKGREEN,
                              "------------------------------------------------------------------------------------------------------------------",
                              bcolors.ENDC)
                        break
                else:
                    raise ValueError("Matrix is singular, cannot find its inverse.")

            # Zero out the elements below\above the diagonal
            for k in range(i, n) if direction == 1 else range(i, -1, -1):
                # Flag: False for summation, True for multiplication
                flag = False

                if i == k:
                    # Calculate the scalar of the diagonal element for normalization to '1'
                    if matrix[i, i] != 1:
                        scalar = 1.0 / matrix[i, i]
                        flag = True
                    # The diagonal element is already '1' no need to normalize continue to the next interation
                    else:
                        continue
                # Calculate the scalar for zeroing out
                elif i != k and matrix[k, i] != 0:
                    scalar = -matrix[k, i] / matrix[i, i]
                else:
                    raise ValueError("Matrix is singular, cannot find its inverse.")

                e_matrices.append(row_addition_elementary_matrix(n, k, i, scalar))
                e_count += 1

                if flag:
                    print(
                        f"elementary matrix E{e_count} for R{k + 1} = {scalar} * (R{i + 1}):\n {e_matrices[e_count - 1]} \n")
                else:
                    print(
                        f"elementary matrix E{e_count} for R{k + 1} = R{k + 1} + ({scalar}R{i + 1}):\n {e_matrices[e_count - 1]} \n")

                matrix = MultiplyMatrix(e_matrices[e_count - 1], matrix)
                print(bcolors.YELLOW, f"The matrix after elementary operation :\n {matrix}", bcolors.ENDC)
                print(bcolors.OKGREEN,"------------------------------------------------------------------------------------------------------------------",bcolors.ENDC)
                inverse_mat = MultiplyMatrix(e_matrices[e_count - 1], inverse_mat)

    return inverse_mat

if __name__ == '__main__':
    A = np.array([[1, 2, 3], [3, 4, 5], [5, 7, 8]])
    # A = np.array([[1, 2, 3], [3, 5, 6], [7, 8, 9,]])
    # A = np.array([[1, 2, 3], [3, 0, 6], [7, 0, 9]])
    # A = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [8, 8, 8, 8], [24, 15, 22, 1]])
    # A = np.array([[2, 1.7, -2.5],
    #               [1.24, -2, -0.5],
    #               [3, 0.2, 1]])

    try:
        A_inverse = inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print(
            "=====================================================================================================================",
            bcolors.ENDC)

    except ValueError as e:
        print(bcolors.FAIL, str(e), bcolors.ENDC)
