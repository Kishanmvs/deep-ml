import math

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d
    determinant = a * d - b * c

    discriminant = trace**2 - 4 * determinant

    sqrt_d = math.sqrt(max(0, discriminant))

    eigenvalues = [
        (trace + sqrt_d) / 2,
        (trace - sqrt_d) / 2
    ]

    return sorted(eigenvalues, reverse=True)