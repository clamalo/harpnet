# DIRECTORIES
base_dir = '/Volumes/T9/'
nc_dir = f'{base_dir}monthly/'
domains_dir = f'{base_dir}domains/'
checkpoints_dir = f'{base_dir}exp_checkpoints/'
figures_dir = f'/Users/clamalo/documents/harpnet/figures/'

# COMPUTING CONTROLS
device = 'mps'
training_batch_size = 32
operational_batch_size = 16

# GRID CONTROLS
scale_factor = 4
start_lat, start_lon = 30, -125
end_lat, end_lon = 51, -104