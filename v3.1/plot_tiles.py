# File: /plot_tiles.py

from config import (
    FIGURES_DIR
)
from tiles import get_tile_dict
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

def plot_tiles(plot_secondary_separately=False):
    """
    Plot all the tiles in the dataset, using the function(s) in tiles.py to retrieve tile boundaries.
    
    1. If plot_secondary_separately=False, both primary and secondary tiles will be plotted on the
       same subplot. Primary tiles are shown in red (solid lines), secondary tiles in blue (dashed lines).
    
    2. If plot_secondary_separately=True, two subplots are created: one for primary tiles (red, solid)
       and one for secondary tiles (blue, dashed).
    
    3. At the center of each tile, place the tile number in the color corresponding to its tile type.
       A white stroke is drawn around the text to improve readability.
    
    4. The plot uses a Cartopy PlateCarree projection with state borders, and its extent is buffered
       by 1 degree around the tiles. The state lines and other basemap features are drawn first, then
       the tile lines are plotted on top.
    
    5. The figure is saved to FIGURES_DIR / 'tiles.png'.
    """
    # Retrieve the tiles from our tile dictionary
    tile_dict = get_tile_dict()

    # Separate primary vs. secondary tiles
    primary_tiles = []
    secondary_tiles = []
    for tid, (min_lat, max_lat, min_lon, max_lon, t_type) in tile_dict.items():
        if t_type == 'primary':
            primary_tiles.append((tid, min_lat, max_lat, min_lon, max_lon))
        else:
            secondary_tiles.append((tid, min_lat, max_lat, min_lon, max_lon))

    # Determine bounding box of all tiles (primary + secondary), add a 1 degree buffer
    all_lats = [t[1] for t in primary_tiles] + [t[2] for t in primary_tiles] + \
               [t[1] for t in secondary_tiles] + [t[2] for t in secondary_tiles]
    all_lons = [t[3] for t in primary_tiles] + [t[4] for t in primary_tiles] + \
               [t[3] for t in secondary_tiles] + [t[4] for t in secondary_tiles]

    if not all_lats or not all_lons:
        # Edge case: in the extremely unlikely scenario no tiles exist, just print a message
        print("No tiles to plot.")
        return

    min_lat = min(all_lats) - 1
    max_lat = max(all_lats) + 1
    min_lon = min(all_lons) - 1
    max_lon = max(all_lons) + 1

    def _plot_tile_list(ax, tile_list, color, linestyle, linewidth=2.0, fontsize=14, rect_zorder=2):
        """
        Helper function to plot a list of tile boundaries and text labels.
        The zorder is set higher than the basemap so these lines/text appear on top.
        """
        for tid, min_lt, max_lt, min_ln, max_ln in tile_list:
            # Draw rectangle
            rect = Rectangle(
                xy=(min_ln, min_lt),
                width=(max_ln - min_ln),
                height=(max_lt - min_lt),
                edgecolor=color,
                facecolor='none',
                linewidth=linewidth,
                linestyle=linestyle,
                transform=ccrs.PlateCarree(),
                zorder=rect_zorder
            )
            ax.add_patch(rect)

            # Label tile in its center; use a white stroke for better readability
            center_lat = 0.5 * (min_lt + max_lt)
            center_lon = 0.5 * (min_ln + max_ln)
            ax.text(
                center_lon,
                center_lat,
                str(tid),
                color=color,
                fontsize=fontsize,
                ha='center',
                va='center',
                transform=ccrs.PlateCarree(),
                zorder=rect_zorder + 1,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')]
            )

    if plot_secondary_separately:
        # Two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(16, 8),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Set extent for both subplots
        ax1.set_extent([min_lon, max_lon, min_lat, max_lat], ccrs.PlateCarree())
        ax2.set_extent([min_lon, max_lon, min_lat, max_lat], ccrs.PlateCarree())

        # Draw basemap first, with a lower zorder
        for ax in (ax1, ax2):
            ax.add_feature(cfeature.STATES, edgecolor='black', zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=1)

        # Plot primary on ax1, secondary on ax2, using higher zorders
        _plot_tile_list(ax1, primary_tiles, color='red',   linestyle='-',  rect_zorder=2)
        _plot_tile_list(ax2, secondary_tiles, color='blue', linestyle='--', rect_zorder=2)

        ax1.set_title("Primary Tiles", fontsize=16)
        ax2.set_title("Secondary Tiles", fontsize=16)

    else:
        # Single subplot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], ccrs.PlateCarree())

        # Draw basemap first, with a lower zorder
        ax.add_feature(cfeature.STATES, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=1)

        # Plot primary (red, solid), then secondary (blue, dashed), both with higher zorders
        _plot_tile_list(ax, primary_tiles, color='red',   linestyle='-',  rect_zorder=2)
        _plot_tile_list(ax, secondary_tiles, color='blue', linestyle='-', rect_zorder=2)

        ax.set_title("Primary & Secondary Tiles Together", fontsize=16)

    # Save the figure
    out_path = FIGURES_DIR / "tiles.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Tile plot saved to {out_path}")

if __name__ == "__main__":
    plot_secondary_separately = True
    plot_tiles(plot_secondary_separately)