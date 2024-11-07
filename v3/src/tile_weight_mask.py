import numpy as np
import matplotlib.pyplot as plt

def tile_weight_mask(N=64):

    step_increment = 1/(N-2)

    """
    Creates an NxN NumPy array where the four central cells have a value of 0,
    and each concentric layer outward increments the value by 1 based on Manhattan distance.
    
    Parameters:
    - N (int): Size of the grid (must be even).
    
    Returns:
    - grid (np.ndarray): The resulting NxN grid with distance values.
    - steps (int): Number of steps to reach the central cells from the farthest cell.
    
    The function also plots the grid.
    """
    if N % 2 != 0:
        raise ValueError("N must be an even number.")
    
    # Define central indices
    c1 = N // 2 - 1
    c2 = N // 2
    
    # Generate grid indices
    rows, cols = np.indices((N, N))
    
    # Calculate Manhattan distance to each of the four central cells
    dist1 = np.abs(rows - c1) + np.abs(cols - c1)
    dist2 = np.abs(rows - c1) + np.abs(cols - c2)
    dist3 = np.abs(rows - c2) + np.abs(cols - c1)
    dist4 = np.abs(rows - c2) + np.abs(cols - c2)
    
    # Take the minimum distance to any of the four central cells
    grid = np.minimum(np.minimum(dist1, dist2), np.minimum(dist3, dist4))
    
    # Number of steps is the maximum distance
    grid = 1-(grid*step_increment)
    
    return grid


# Example usage:
if __name__ == "__main__":
    N = 64  # Example even number
    grid = tile_weight_mask(N)
    print("Distance Grid:\n", grid)