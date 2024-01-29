import numpy as np

def make_blobs(blob0: tuple, blob1: tuple) -> tuple:
    """
    Parameters
    ----------
    blob0: tuple[(mean_x, mean_y), (std_x, std_y), size]
    blob1: tuple[(mean_x, mean_y), (std_x, std_y), size]

    Returns
    -------
    tuple[blob0: tuple[x, y], blob1: tuple[x, y]]
    """
    
    blob0_x = np.random.normal(blob0[0][0], blob0[1][0], blob0[2])
    blob0_y = np.random.normal(blob0[0][1], blob0[1][1], blob0[2])

    blob1_x = np.random.normal(blob1[0][0], blob1[1][0], blob1[2])
    blob1_y = np.random.normal(blob1[0][1], blob1[1][1], blob1[2])

    return np.vstack((blob0_x, blob0_y)).T, np.vstack((blob1_x, blob1_y)).T

def unbiased_var(x):
    ux = np.mean(x)
    return np.sum([(_x - ux) ** 2 for _x in x]) / (len(x) - 1)

def t_statistic(x: np.ndarray, y: np.ndarray):
    ux, uy = np.mean(x), np.mean(y)
    vx, vy = unbiased_var(x), unbiased_var(y)
    s = np.sqrt(
        (((x.size - 1) * vx) + ((y.size - 1) * vy)) / (x.size + y.size - 2)
    )
    return (ux - uy) / (s * np.sqrt((1 / x.size) + (1 / y.size)))

