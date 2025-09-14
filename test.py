import numpy as np

A = np.matrix([
    [0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0]
])

A = np.matrix([
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0]
])

Asqr = A @ A
Acube = A @ A @ A @ A
print(Acube)
print('\n', Asqr)

def numpy_to_latex_bmatrix(matrix: np.ndarray) -> str:
    if len(matrix.shape) != 2:
        raise ValueError("Input must be a 2D matrix")
    latex = "\\begin{bmatrix}\n"
    for row in matrix:
        for elem in row:
            row_str = " & ".join(map(str, elem))
            latex += f"  {row_str} \\\\\n"
    latex += "\\end{bmatrix}"
    return latex

mat = numpy_to_latex_bmatrix(Acube)
# print(mat)