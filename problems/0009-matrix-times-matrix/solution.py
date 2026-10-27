def matrixmul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]] | int:

    # Check if matrix dimensions are compatible
    if len(a[0]) != len(b):
        return -1

    # Initialize result matrix with zeros
    rows = len(a)
    cols = len(b[0])
    c = [[0] * cols for _ in range(rows)]

    # Matrix multiplication
    for i in range(rows):
        for j in range(cols):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]

    return c