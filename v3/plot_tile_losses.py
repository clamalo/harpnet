# File: plot_tile_losses.py

import math
import torch
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib import colors

from config import CHECKPOINTS_DIR, FIGURES_DIR
from tiles import get_tile_dict

def plot_tile_losses():
    """
    Plot each tile (both primary and secondary) on a Cartopy map, color‑coded by its
    test loss. Loads tile-specific best checkpoints from CHECKPOINTS_DIR / 'best' /
    '<tile_id>_best.pt', reading the 'test_loss' from each.

    Saves the resulting figure to FIGURES_DIR / 'tile_test_losses.png'.
    """

    # -------------------------------------------------------------------
    # 1) Gather tile boundaries from get_tile_dict()
    # -------------------------------------------------------------------
    tile_dict = get_tile_dict()
    if not tile_dict:
        print("No tiles found in tile_dict. Exiting.")
        return

    # -------------------------------------------------------------------
    # 2) For each tile, look for <tile_id>_best.pt and read test_loss
    # -------------------------------------------------------------------
    tile_losses = {}
    for tile_id, (min_lat, max_lat, min_lon, max_lon, tile_type) in tile_dict.items():
        ckpt_path = CHECKPOINTS_DIR / "best" / f"{tile_id}_best.pt"
        if ckpt_path.exists():
            try:
                ckpt_data = torch.load(ckpt_path, map_location="cpu")
                test_loss = ckpt_data.get('test_loss', math.inf)
                tile_losses[tile_id] = test_loss
            except Exception as e:
                print(f"Could not load checkpoint for tile {tile_id}: {e}")
                tile_losses[tile_id] = math.inf
        else:
            # If no checkpoint, we skip or assign inf
            tile_losses[tile_id] = math.inf

    # Filter out tiles with no valid checkpoint (test_loss = inf)
    # (If you want to show them in a special color, comment out the filtering.)
    # However, we can keep them and color them differently if desired:
    # tile_losses = {tid: loss for tid, loss in tile_losses.items() if loss < math.inf}

    # -------------------------------------------------------------------
    # 3) Build a colormap based on [min_loss, max_loss] among loaded tiles
    # -------------------------------------------------------------------
    # If *all* are inf or if no valid tile was found, handle gracefully:
    valid_losses = [loss for loss in tile_losses.values() if loss < math.inf]
    if len(valid_losses) == 0:
        print("No valid tile-specific checkpoints found (all test_loss = inf).")
        return

    min_loss = min(valid_losses)
    max_loss = max(valid_losses)
    norm = colors.Normalize(vmin=min_loss, vmax=max_loss)
    cmap = cm.viridis  # Choose your favorite colormap (e.g., 'plasma', 'coolwarm', etc.)

    # -------------------------------------------------------------------
    # 4) Prepare a Cartopy figure, plot each tile with a colored face
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Determine bounding box of all tiles (with a small buffer)
    all_lats = [v[0] for v in tile_dict.values()] + [v[1] for v in tile_dict.values()]
    all_lons = [v[2] for v in tile_dict.values()] + [v[3] for v in tile_dict.values()]
    buffer = 1.0
    min_lat, max_lat = min(all_lats) - buffer, max(all_lats) + buffer
    min_lon, max_lon = min(all_lons) - buffer, max(all_lons) + buffer
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], ccrs.PlateCarree())

    # Draw some geographical features
    ax.add_feature(cfeature.STATES, edgecolor='black', zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=1)

    for tile_id, (min_lt, max_lt, min_ln, max_ln, tile_type) in tile_dict.items():
        loss_val = tile_losses.get(tile_id, math.inf)
        # Color for this tile
        if loss_val == math.inf:
            # fallback color for unknown
            facecolor = "lightgray"
        else:
            facecolor = cmap(norm(loss_val))

        rect = Rectangle(
            xy=(min_ln, min_lt),
            width=(max_ln - min_ln),
            height=(max_lt - min_lt),
            edgecolor='black',
            facecolor=facecolor,
            linewidth=1,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=2
        )
        ax.add_patch(rect)

        # Label tile in its center, with white stroke around text
        center_lon = 0.5 * (min_ln + max_ln)
        center_lat = 0.5 * (min_lt + max_lt)
        label_str = (
            f"{tile_id}\n"
            f"loss={loss_val:.4f}" if loss_val < math.inf else f"{tile_id}\n(no ckpt)"
        )
        # ax.text(
        #     center_lon,
        #     center_lat,
        #     label_str,
        #     color='white',
        #     fontsize=8,
        #     ha='center',
        #     va='center',
        #     transform=ccrs.PlateCarree(),
        #     zorder=3,
        #     path_effects=[pe.withStroke(linewidth=2, foreground='black')]
        # )

    # -------------------------------------------------------------------
    # 5) Create a colorbar
    # -------------------------------------------------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Not used for actual data plotting, just needed for colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label("Test Loss")

    ax.set_title("Tile-Specific Test Losses", fontsize=16)

    # Save figure
    out_path = FIGURES_DIR / "tile_test_losses.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Tile test-loss map saved to {out_path}")


if __name__ == "__main__":
    plot_tile_losses()