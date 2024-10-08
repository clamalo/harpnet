import torch.nn as nn
import torch
import numpy as np
import os
os.system('ulimit -n 1024')  # Set the maximum number of open file descriptors to 1024
import sys
import warnings
warnings.filterwarnings("ignore")  # Ignore any warning messages
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set up the current working directory
cwd_dir = (os.path.abspath(os.path.join(os.getcwd())))
sys.path.insert(0, cwd_dir)

# Import custom modules
from utils.model import UNetWithAttention
from utils.utils3 import *
import utils.constants as constants


def setup(domain=None):
    # Create directories if they don't exist
    if not os.path.exists(f'{constants.domains_dir}'):
        os.makedirs(f'{constants.domains_dir}')
    if domain is not None:
        if not os.path.exists(f'{constants.domains_dir}/{domain}/'):
            os.makedirs(f'{constants.domains_dir}/{domain}/')
    if not os.path.exists(f'{constants.checkpoints_dir}'):
        os.makedirs(f'{constants.checkpoints_dir}')
    # Create utils/data directory
    if not os.path.exists(f'utils/data/'):
        os.makedirs(f'utils/data/')
    # Create figure directories
    if not os.path.exists(f'{constants.figures_dir}'):
        os.makedirs(f'{constants.figures_dir}')
    if not os.path.exists(f'{constants.figures_dir}train/'):
        os.makedirs(f'{constants.figures_dir}train/')
    if not os.path.exists(f'{constants.figures_dir}test/'):
        os.makedirs(f'{constants.figures_dir}test/')
    if not os.path.exists(f'{constants.figures_dir}stitch/'):
        os.makedirs(f'{constants.figures_dir}stitch/')


def create_grid_domains():
    # Create a grid of domains based on latitude and longitude ranges
    start_lat, start_lon = constants.start_lat, constants.start_lon
    end_lat, end_lon = constants.end_lat, constants.end_lon
    grid_domains = {}
    total_domains = 0
    # Loop through latitude and longitude to create grid cells
    for lat in range(start_lat, end_lat, 4):
        for lon in range(start_lon, end_lon, 4):
            grid_domains[total_domains] = [lat, lat + 4, lon, lon + 4]
            total_domains += 1
    # Save grid domains to a pickle file
    with open(f'{constants.domains_dir}grid_domains.pkl', 'wb') as f:
        pickle.dump(grid_domains, f)


def xr_to_np(domain, first_month, last_month, pad=False):
    # Convert xarray datasets to numpy arrays
    first_month = datetime(first_month[0], first_month[1], 1)
    last_month = datetime(last_month[0], last_month[1], 1)
    # Load grid domain information
    with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)
    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]

    # Load reference dataset
    reference_ds = xr.load_dataset(f'utils/reference_ds.grib2', engine='cfgrib')
    # Adjust longitudes and sort by latitude and longitude
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    reference_ds = reference_ds.sortby('latitude', ascending=True)

    # Calculate total number of months to process
    total_months = (last_month.year - first_month.year) * 12 + last_month.month - first_month.month + 1

    current_month = first_month
    for _ in tqdm(range(total_months), desc="Processing months"):
        year, month = current_month.year, current_month.month
        # Load monthly dataset
        ds = xr.open_dataset(f'{constants.nc_dir}{year}-{month:02d}.nc')
        # Filter times to specific hours
        time_index = pd.DatetimeIndex(ds.time.values)
        filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
        ds = ds.sel(time=filtered_times)
        ds = ds.sortby('time')
        ds['days'] = ds.time.dt.dayofyear

        # Crop reference dataset to the domain's latitude and longitude range
        cropped_reference_ds = reference_ds.sel(latitude=slice(min_lat, max_lat-0.25), longitude=slice(min_lon, max_lon-0.25))
        cropped_reference_ds_latitudes = cropped_reference_ds.latitude.values
        cropped_reference_ds_longitudes = cropped_reference_ds.longitude.values
        if pad:
            # Crop input reference dataset with padding if required
            cropped_input_reference_ds = reference_ds.sel(latitude=slice(min_lat-0.25, max_lat), longitude=slice(min_lon-0.25, max_lon))
            cropped_input_reference_ds_latitudes = cropped_input_reference_ds.latitude.values
            cropped_input_reference_ds_longitudes = cropped_input_reference_ds.longitude.values
        
        # Scale coordinates to finer resolution
        fine_lats = scale_coordinates(cropped_reference_ds_latitudes, constants.scale_factor)
        fine_lons = scale_coordinates(cropped_reference_ds_longitudes, constants.scale_factor)

        # Interpolate the dataset to the fine resolution
        fine_ds = ds.interp(lat=fine_lats, lon=fine_lons)
        if pad:
            # Interpolate with padding
            coarse_ds = ds.interp(lat=cropped_input_reference_ds_latitudes, lon=cropped_input_reference_ds_longitudes)
        else:
            # Interpolate without padding
            coarse_ds = ds.interp(lat=cropped_reference_ds_latitudes, lon=cropped_reference_ds_longitudes)

        # Save input, target, and time data to numpy files
        times = ds.time.values
        np.save(f'{constants.domains_dir}{domain}/input_{year}_{month:02d}.npy', coarse_ds.tp.values.astype('float32'))
        np.save(f'{constants.domains_dir}{domain}/target_{year}_{month:02d}.npy', fine_ds.tp.values.astype('float32'))
        np.save(f'{constants.domains_dir}{domain}/times_{year}_{month:02d}.npy', times)
        current_month += relativedelta(months=1)

    
def generate_dataloaders(domain, first_month, last_month, train_test):
    # Generate dataloaders for training and testing
    input_file_paths = []
    target_file_paths = []
    times_file_paths = []
    first_month = datetime(first_month[0], first_month[1], 1)
    last_month = datetime(last_month[0], last_month[1], 1)
    current_month = first_month
    # Collect file paths for input, target, and time data
    while current_month <= last_month:
        input_fp = f'{constants.domains_dir}{domain}/input_{current_month.year}_{current_month.month:02d}.npy'
        target_fp = f'{constants.domains_dir}{domain}/target_{current_month.year}_{current_month.month:02d}.npy'
        times_fp = f'{constants.domains_dir}{domain}/times_{current_month.year}_{current_month.month:02d}.npy'
        input_file_paths.append(input_fp)
        target_file_paths.append(target_fp)
        times_file_paths.append(times_fp)
        current_month += relativedelta(months=1)

    # Load and concatenate all data
    input_arr = np.concatenate([np.load(fp) for fp in input_file_paths])
    target_arr = np.concatenate([np.load(fp) for fp in target_file_paths])
    times_arr = np.concatenate([np.load(fp) for fp in times_file_paths])

    # Convert times to float for sorting
    times_arr = times_arr.astype('datetime64[s]').astype(np.float64)

    # Shuffle data
    np.random.seed(42)
    indices = np.argsort(times_arr)
    np.random.shuffle(indices)
    input_arr = input_arr[indices]
    target_arr = target_arr[indices]
    times_arr = times_arr[indices]

    print(times_arr[:3].astype('datetime64[s]'))

    # Split data into training and testing sets
    test_input_arr, train_input_arr = np.split(input_arr, [int(train_test * len(input_arr))])
    test_target_arr, train_target_arr = np.split(target_arr, [int(train_test * len(target_arr))])
    test_times_arr, train_times_arr = np.split(times_arr, [int(train_test * len(times_arr))])

    # Create datasets and dataloaders
    train_dataset = MemMapDataset(train_input_arr, train_target_arr, train_times_arr)
    test_dataset = MemMapDataset(test_input_arr, test_target_arr, test_times_arr)

    torch.manual_seed(42)
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=constants.training_batch_size, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=constants.training_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def train(domain, model, dataloader, criterion, optimizer, device, pad=False, plot=False):
    # Training loop for the model
    model.train()
    losses = []

    if plot:
        lats, lons, input_lats, input_lons = get_lats_lons(domain, pad)
        random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
        plotted = 0

    for i, (inputs, targets, times) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Zero the gradients

        output_shape = (64, 64)
        interpolated_inputs = inputs.unsqueeze(1)  # Add channel dimension
        interpolated_inputs = nn.functional.interpolate(interpolated_inputs, size=output_shape, mode=interpolation_method)  # Interpolate input to output shape

        outputs = model(inputs)  # Forward pass

        loss = criterion(outputs, targets)  # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        losses.append(loss.item())

        # Plot some random samples if needed
        if plot and i in random_10:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
            axs[0].pcolormesh(input_lons, input_lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[1].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[2].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            for ax in axs:
                ax.coastlines()
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
            box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                    linewidth=1, edgecolor='r', facecolor='none')
            axs[0].add_patch(box)
            plt.suptitle(f'Loss: {criterion(outputs[0], targets[0]).item():.3f}')
            plt.savefig(f'{constants.figures_dir}train/{plotted}.png')
            plt.close()
            plotted += 1

    return np.mean(losses)


def test(domain, model, dataloader, criterion, device, pad=False, plot=True):
    # Testing loop for the model
    model.eval()
    losses = []
    bilinear_losses = []

    gridded_losses = []
    gridded_bilinear_losses = []

    if plot:
        lats, lons, input_lats, input_lons = get_lats_lons(domain, pad)
        random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
        plotted = 0

    for i, (inputs, targets, times) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        output_shape = (64, 64)
        interpolated_inputs = inputs.unsqueeze(1)  # Add channel dimension
        interpolated_inputs = nn.functional.interpolate(interpolated_inputs, size=output_shape, mode=interpolation_method)  # Interpolate input to output shape

        with torch.no_grad():
            outputs = model(inputs)  # Forward pass

        loss = criterion(outputs, targets)  # Calculate loss
        gridded_loss = ((outputs - targets) ** 2).cpu().detach().numpy()
        
        # Crop inputs and interpolate using bilinear method for comparison
        cropped_inputs = inputs[:, 1:-1, 1:-1]
        interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
        
        bilinear_loss = criterion(interpolated_inputs, targets)  # Calculate bilinear loss
        gridded_bilinear_loss = ((interpolated_inputs - targets) ** 2).cpu().detach().numpy()

        # Regular losses
        losses.append(loss.item())
        bilinear_losses.append(bilinear_loss.item())

        # Gridded losses
        gridded_losses.append(gridded_loss.mean(axis=0))
        gridded_bilinear_losses.append(gridded_bilinear_loss.mean(axis=0))

        # Plot some random samples if needed
        if plot and i in random_10:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
            axs[0].pcolormesh(input_lons, input_lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[1].pcolormesh(lons, lats, interpolated_inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[2].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[3].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            for ax in axs:
                ax.coastlines()
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
            box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                    linewidth=1, edgecolor='r', facecolor='none')
            axs[0].add_patch(box)
            axs[0].set_title('Input')
            axs[1].set_title('Bilinear')
            axs[2].set_title('HARPNET Output')
            axs[3].set_title('Target')
            plt.suptitle(f'Loss: {criterion(outputs[0], targets[0]).item():.3f}')
            plt.savefig(f'{constants.figures_dir}test/{plotted}.png')
            plt.close()
            plotted += 1

    gridded_losses = np.array(gridded_losses)
    gridded_bilinear_losses = np.array(gridded_bilinear_losses)
    mean_losses = gridded_losses.mean(axis=0)
    mean_bilinear_losses = gridded_bilinear_losses.mean(axis=0)
    max_value = max(mean_losses.max(), mean_bilinear_losses.max())
    max_value = 1.75

    import cartopy.crs as ccrs
    fine_lats, fine_lons, _, _ = get_lats_lons(domain, pad=True)

    #plot the mean_losses with cartopy states 
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]])
    ax.set_title('Mean Losses')
    cf = ax.imshow(mean_losses, origin='upper', extent=(fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]), transform=ccrs.PlateCarree(), vmin=0, vmax=max_value)
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='Mean Losses')
    plt.savefig('utah.png')

    #plot the mean_bilinear_losses with cartopy states 
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]])
    ax.set_title('Mean Losses')
    cf = ax.imshow(mean_bilinear_losses, origin='upper', extent=(fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]), transform=ccrs.PlateCarree(), vmin=0, vmax=max_value)
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='Mean Losses')
    plt.savefig('utah_bilinear.png')

    return np.mean(losses), np.mean(bilinear_losses)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels):
        # Attention block to capture important features
        super(AttentionBlock, self).__init__()
        self.theta_x = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(2, 2))
        self.phi_g = nn.Conv2d(in_channels=gating_channels, out_channels=in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.psi = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, gating):
        # Forward pass for attention block
        theta_x = self.theta_x(x)  # Downsample input feature map
        gating = self.phi_g(gating)  # Apply gating
        add = self.relu(theta_x + gating)  # Add and apply ReLU activation
        psi = self.psi(add)  # Apply psi convolution
        sigmoid_psi = self.sigmoid(psi)  # Apply sigmoid to get attention map
        upsample_psi = self.upsample(sigmoid_psi)  # Upsample the attention map
        y = upsample_psi * x  # Apply attention to input feature map
        return y
    

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shape=(64,64), dropout_rate=0.0):
        # Residual convolutional block with optional dropout
        super(ResConvBlock, self).__init__()
        self.resconvblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),  # Apply layer normalization
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LayerNorm([out_channels, shape[0], shape[1]]),
            nn.ReLU())
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),  # Shortcut connection
            nn.LayerNorm([out_channels, shape[0], shape[1]]))
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
    
    def forward(self, x):
        # Forward pass for residual block
        x_shortcut = self.shortcut(x)  # Shortcut connection
        x = self.resconvblock(x)  # Residual path
        x = x + x_shortcut  # Add shortcut to residual
        x = self.dropout(x)  # Apply dropout
        return x
    
    
class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape=(64, 64)):
        # UNet model with attention blocks
        super(UNetWithAttention, self).__init__()

        # Encoding layers
        self.enc1 = ResConvBlock(in_channels, 64, (64,64))
        self.enc2 = ResConvBlock(64, 128, (32,32))
        self.enc3 = ResConvBlock(128, 256, (16,16))
        self.enc4 = ResConvBlock(256, 512, (8,8))
        self.enc5 = ResConvBlock(512, 1024, (4,4))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer

        # Bridge layer
        self.bridge = ResConvBlock(1024, 2048, (2,2))

        # Attention blocks
        self.attn_block5 = AttentionBlock(1024, 2048)
        self.attn_block4 = AttentionBlock(512, 1024)
        self.attn_block3 = AttentionBlock(256, 512)
        self.attn_block2 = AttentionBlock(128, 256)
        self.attn_block1 = AttentionBlock(64, 128)

        # Upsampling and decoding layers
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Decoding layers with residual connections
        self.dec5 = ResConvBlock(2048, 1024, (4,4), dropout_rate=0.5)
        self.dec4 = ResConvBlock(1024, 512, (8,8), dropout_rate=0.5)
        self.dec3 = ResConvBlock(512, 256, (16,16), dropout_rate=0.3)
        self.dec2 = ResConvBlock(256, 128, (32,32), dropout_rate=0.3)
        self.dec1 = ResConvBlock(128, 64, (64,64), dropout_rate=0.1)

        # Final convolution layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.output_shape = output_shape

    def forward(self, x):
        # Forward pass for UNet with attention
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing

        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Bridge
        bridge = self.bridge(self.pool(enc5))
        
        # Decoding path with attention blocks
        gating5 = self.attn_block5(enc5, bridge)
        up5 = self.upconv5(bridge)
        up5 = torch.cat([up5, gating5], dim=1)
        dec5 = self.dec5(up5)

        gating4 = self.attn_block4(enc4, dec5)
        up4 = self.upconv4(dec5)
        up4 = torch.cat([up4, gating4], dim=1)
        dec4 = self.dec4(up4)

        gating3 = self.attn_block3(enc3, dec4)
        up3 = self.upconv3(dec4)
        up3 = torch.cat([up3, gating3], dim=1)
        dec3 = self.dec3(up3)

        gating2 = self.attn_block2(enc2, dec3)
        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, gating2], dim=1)
        dec2 = self.dec2(up2)

        gating1 = self.attn_block1(enc1, dec2)
        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, gating1], dim=1)
        dec1 = self.dec1(up1)

        # Final output
        final = self.final_conv(dec1)
        
        # Clamp output to non-negative values
        final = torch.clamp(final, min=0)

        return final.squeeze(1)
    


# Main execution block

# Define domains to process
domains = [15]

setup()  # Set up directories
for domain in domains:
    LOAD = False
    first_month = (1979, 10)
    last_month = (1980, 9)
    train_test = 0.2  # Train-test split ratio
    continue_epoch = False
    max_epoch = 5
    pad = True
    interpolation_method = 'nearest'

    if LOAD:
        setup(domain)
        create_grid_domains()  # Create grid domains
        xr_to_np(domain, first_month, last_month, pad=pad)  # Convert xarray to numpy arrays

    # Generate dataloaders for training and testing
    train_dataloader, test_dataloader = generate_dataloaders(domain, first_month, last_month, train_test)

    # Print data shapes for debugging
    print(len(train_dataloader), next(iter(train_dataloader))[0].numpy().shape, next(iter(train_dataloader))[1].numpy().shape)
    print(len(test_dataloader), next(iter(test_dataloader))[0].numpy().shape, next(iter(test_dataloader))[1].numpy().shape)

    # Initialize model, optimizer, and loss function
    model = UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Load checkpoint if continuing training
    if continue_epoch:
        checkpoint = torch.load(f'{constants.checkpoints_dir}{domain}/{continue_epoch-1}_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Train and test the model
    for epoch in range(continue_epoch or 0, max_epoch):
        train_loss = train(domain, model, train_dataloader, criterion, optimizer, constants.device, pad=pad, plot=False)
        test_loss, bilinear_loss = test(domain, model, test_dataloader, criterion, constants.device, pad=pad, plot=False)
        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Bilinear Loss: {bilinear_loss:.4f}')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_loss': bilinear_loss
        }
        # Make sure checkpoint directory exists
        os.makedirs(f'{constants.checkpoints_dir}{domain}', exist_ok=True)
        # torch.save(checkpoint, f'{constants.checkpoints_dir}{domain}/{epoch}_model.pt')