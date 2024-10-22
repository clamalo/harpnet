raw_dir = f'/Volumes/seagate/monthly'
processed_dir = f'/Users/clamalo/documents/harpnet/v2/tiles'
checkpoints_dir = f'/Volumes/seagate/v2_checkpoints'
figures_dir = f'figures'

torch_device = 'mps'

# GRID CONTROLS
scale_factor = 8  # 8 for 3km, 4 for 6km
min_lat, min_lon = 30, -125
max_lat, max_lon = 51, -104