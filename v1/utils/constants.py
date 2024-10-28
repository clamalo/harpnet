# DIRECTORIES
base_dir = '/Volumes/seagate/'
nc_dir = f'{base_dir}monthly/'
# domains_dir = f'{base_dir}domains/'
domains_dir = f'/Users/clamalo/documents/harpnet/domains/'
# checkpoints_dir = f'{base_dir}exp_checkpoints/'
checkpoints_dir = f'/Users/clamalo/documents/harpnet/checkpoints/'
figures_dir = f'/Users/clamalo/documents/harpnet/figures/'

# COMPUTING CONTROLS
device = 'mps'
training_batch_size = 32
operational_batch_size = 16

# GRID CONTROLS
scale_factor = 4
start_lat, start_lon = 30, -125
end_lat, end_lon = 51, -104