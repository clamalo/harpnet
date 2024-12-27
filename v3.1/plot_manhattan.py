import numpy as np
import matplotlib.pyplot as plt

def tile_weight_mask(N=64):
    """
    Create an N×N NumPy array of Manhattan-based weights.
    The four central cells are ~1, and edges approach 0.
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")

    # Indices of the 2×2 center
    c1 = N // 2 - 1
    c2 = N // 2

    # Grid of row/col indices
    rows, cols = np.indices((N, N))

    # Compute Manhattan distance to each of the 4 central cells
    dist1 = np.abs(rows - c1) + np.abs(cols - c1)
    dist2 = np.abs(rows - c1) + np.abs(cols - c2)
    dist3 = np.abs(rows - c2) + np.abs(cols - c1)
    dist4 = np.abs(rows - c2) + np.abs(cols - c2)

    # Minimum distance to any center cell
    dist = np.minimum(np.minimum(dist1, dist2), np.minimum(dist3, dist4))
    max_dist = dist.max()

    # Linear flip so distance=0 => weight=1, distance=max => weight=0
    mask = 1.0 - (dist / max_dist)
    return mask.astype(np.float32)

if __name__ == "__main__":
    N = 64
    mask = tile_weight_mask(N)
    plt.imshow(mask, origin='lower', cmap='viridis')
    plt.colorbar(label='Manhattan Weight')
    plt.title(f"Manhattan Mask (N={N})")
    plt.show()