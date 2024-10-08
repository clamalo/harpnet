from utils.utils3 import *
from utils.model import UNetWithAttention

import cartopy.crs as ccrs

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


domains = [24,15]
domain = domains[1]

first_month = (1979, 10)
last_month = (2021, 9)
train_test = 0.2


# sort_epochs()
# setup(domain)
# create_grid_domains()
# xr_to_np(domain, first_month, last_month, pad=True)


model = UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.device)
# model.load_state_dict(torch.load(f'{constants.checkpoints_dir}best/{domain}_model.pt')['model_state_dict'])
model.load_state_dict(torch.load(f'{constants.checkpoints_dir}best/{domain}_model.pt')['model_state_dict'])
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')


fine_lats, fine_lons, _, _ = get_lats_lons(domain, pad=True)


def generate_dataloaders(domain, first_month, last_month, train_test):
    input_file_paths = []
    target_file_paths = []
    times_file_paths = []
    first_month = datetime(first_month[0], first_month[1], 1)
    last_month = datetime(last_month[0], last_month[1], 1)
    current_month = first_month
    while current_month <= last_month:
        input_fp = f'{constants.domains_dir}{domain}/input_{current_month.year}_{current_month.month:02d}.npy'
        target_fp = f'{constants.domains_dir}{domain}/target_{current_month.year}_{current_month.month:02d}.npy'
        times_fp = f'{constants.domains_dir}{domain}/times_{current_month.year}_{current_month.month:02d}.npy'
        input_file_paths.append(input_fp)
        target_file_paths.append(target_fp)
        times_file_paths.append(times_fp)
        current_month += relativedelta(months=1)

    input_arr = np.concatenate([np.load(fp) for fp in input_file_paths])
    target_arr = np.concatenate([np.load(fp) for fp in target_file_paths])
    times_arr = np.concatenate([np.load(fp) for fp in times_file_paths])

    times_arr = times_arr.astype('datetime64[s]').astype(np.float64)

    np.random.seed(42)
    indices = np.argsort(times_arr)
    np.random.shuffle(indices)
    input_arr = input_arr[indices]
    target_arr = target_arr[indices]
    times_arr = times_arr[indices]

    print(times_arr[:3].astype('datetime64[s]'))

    test_input_arr, train_input_arr = np.split(input_arr, [int(train_test * len(input_arr))])
    test_target_arr, train_target_arr = np.split(target_arr, [int(train_test * len(target_arr))])
    test_times_arr, train_times_arr = np.split(times_arr, [int(train_test * len(times_arr))])

    train_dataset = MemMapDataset(train_input_arr, train_target_arr, train_times_arr)
    test_dataset = MemMapDataset(test_input_arr, test_target_arr, test_times_arr)

    torch.manual_seed(42)
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=constants.training_batch_size, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=constants.training_batch_size, shuffle=False)

    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = generate_dataloaders(domain, first_month, last_month, train_test)




losses = []
bilinear_losses = []
for i, (inputs, targets, times) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

    model.eval()

    inputs, targets = inputs.to(constants.device), targets.to(constants.device)
    outputs = model(inputs)

    loss = ((outputs - targets) ** 2).cpu().detach().numpy()

    # compute a bilinear interpolation of the inputs and store it in bilinear_losses
    cropped_inputs = inputs[:, 1:-1, 1:-1]
    interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
    # bilinear_loss = ((interpolated_inputs - targets) ** 2).mean(dim=0).cpu().detach().numpy()
    bilinear_loss = ((interpolated_inputs - targets) ** 2).cpu().detach().numpy()

    losses.append(loss.mean(axis=0))
    bilinear_losses.append(bilinear_loss.mean(axis=0))



losses = np.array(losses)
bilinear_losses = np.array(bilinear_losses)

print(losses.shape)

mean_losses = losses.mean(axis=0)
mean_bilinear_losses = bilinear_losses.mean(axis=0)

max_value = max(mean_losses.max(), mean_bilinear_losses.max())

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