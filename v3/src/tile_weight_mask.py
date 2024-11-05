import numpy as np


def tile_weight_mask(size=64):
    """
    Generates a 2D NumPy array of specified size with:
    - The four central grids set to 1.
    - The corner grids set to 0.
    - The two central grids on each edge set to 0.5.
    - Intermediate grids varying based on their distance from the center.

    Parameters:
    - size (int): The size of the grid (size x size). Default is 64.

    Returns:
    - array (np.ndarray): The generated 2D array.
    """
    # Initialize a size x size array with zeros
    array = np.zeros((size, size))
    
    # Define the center of the array
    center = (size / 2 - 0.5, size / 2 - 0.5)  # e.g., (31.5, 31.5) for size=64
    
    # Create coordinate grids
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the Euclidean distance from each grid to the center
    distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Maximum distance is from center to a corner
    max_distance = np.sqrt((center[0])**2 + (center[1])**2)
    
    # Apply the gradient formula: 1 - (distance / max_distance)^2
    array = 1 - (distance / max_distance)**2
    
    # Ensure that the values are within [0, 1]
    array = np.clip(array, 0, 1)
    
    # Assign exact values to the central grids
    central_indices = [int(center[0]), int(center[1])]
    array[central_indices[0], central_indices[1]] = 1
    array[central_indices[0], central_indices[1]+1] = 1
    array[central_indices[0]+1, central_indices[1]] = 1
    array[central_indices[0]+1, central_indices[1]+1] = 1
    
    # Assign exact values to the corner grids
    corners = [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]
    for corner in corners:
        array[corner] = 0
    
    # Assign exact values to the center of each edge
    edge_centers = [
        (0, int(center[1])), (0, int(center[1]+1)),
        (size-1, int(center[1])), (size-1, int(center[1]+1)),
        (int(center[0]), 0), (int(center[0]+1), 0),
        (int(center[0]), size-1), (int(center[0]+1), size-1)
    ]
    for edge in edge_centers:
        array[edge] = 0.5
    
    return array