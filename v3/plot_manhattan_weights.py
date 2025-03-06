import numpy as np
import matplotlib.pyplot as plt

def tile_weight_mask(N):
    """
    Create an N×N NumPy array of Manhattan-based weights.
    The four central cells are ~1, and the weights decrease toward the edges.
    N must be an even number.
    """
    if N % 2 != 0:
        raise ValueError("N must be an even number.")
    
    c1 = N // 2 - 1
    c2 = N // 2
    rows, cols = np.indices((N, N))
    
    dist1 = np.abs(rows - c1) + np.abs(cols - c1)
    dist2 = np.abs(rows - c1) + np.abs(cols - c2)
    dist3 = np.abs(rows - c2) + np.abs(cols - c1)
    dist4 = np.abs(rows - c2) + np.abs(cols - c2)
    
    dist = np.minimum(np.minimum(dist1, dist2), np.minimum(dist3, dist4))
    max_dist = dist.max()
    mask = 1.0 - (dist / max_dist) if max_dist > 0 else np.ones((N, N), dtype=np.float32)
    return mask.astype(np.float32)

if __name__ == "__main__":
    N = 16  # You can adjust this even number as needed
    weights = tile_weight_mask(N)

    plt.figure(figsize=(6, 6))
    plt.imshow(weights, cmap='viridis', origin='lower')
    plt.title(f"Manhattan Tile Weights (N={N})")
    plt.colorbar(label="Weight")
    plt.tight_layout()
    plt.show()