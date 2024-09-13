import torch.nn as nn
import torch
import numpy as np
import os
os.system('ulimit -n 1024')
import sys
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cwd_dir = (os.path.abspath(os.path.join(os.getcwd())))
sys.path.insert(0, cwd_dir)
from utils.model import UNetWithAttentionMini
from utils.utils import *
import utils.constants as constants


domains = [2]


setup()
for domain in domains:
    LOAD = False
    first_month = (1979, 10)
    last_month = (1980, 9)
    train_test = 0.2
    continue_epoch = False
    max_epoch = 5
    num_members = 3
    pad = True


    if LOAD:
        setup(domain)
        quit()
        create_grid_domains()
        xr_to_np(domain, first_month, last_month, pad=pad)

    train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths = create_paths(domain, first_month, last_month, train_test)


    class MemMapDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __len__(self):
            return self.data.shape[0]
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    

    def create_dataloader(input_file_paths, target_file_paths, batch_size=constants.training_batch_size, shuffle=True):
        def load_files_in_batches(file_paths, batch_size=32):
            arrays = []
            for i in tqdm(range(0, len(file_paths), batch_size)):
                batch_paths = file_paths[i:i + batch_size]
                batch_arrays = [np.load(fp, mmap_mode='r') for fp in batch_paths]
                arrays.append(np.concatenate(batch_arrays, axis=0))
            return np.concatenate(arrays, axis=0)
        input_arr = load_files_in_batches(input_file_paths, batch_size=batch_size)
        target_arr = load_files_in_batches(target_file_paths, batch_size=batch_size)
        dataset = MemMapDataset(input_arr, target_arr)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader


    def create_dataset(input_file_paths, target_file_paths, batch_size=constants.training_batch_size, shuffle=True):
        def load_files_in_batches(file_paths, batch_size=32):
            arrays = []
            for i in tqdm(range(0, len(file_paths), batch_size)):
                batch_paths = file_paths[i:i + batch_size]
                batch_arrays = [np.load(fp, mmap_mode='r') for fp in batch_paths]
                arrays.append(np.concatenate(batch_arrays, axis=0))
            return np.concatenate(arrays, axis=0)
        input_arr = load_files_in_batches(input_file_paths, batch_size=batch_size)
        target_arr = load_files_in_batches(target_file_paths, batch_size=batch_size)
        dataset = MemMapDataset(input_arr, target_arr)
        return dataset


    train_dataset = create_dataset(train_input_file_paths, train_target_file_paths)
    test_dataset, test_dataloader = create_dataloader(test_input_file_paths, test_target_file_paths, shuffle=False)

    from torch.utils.data import DataLoader, SubsetRandomSampler
    def create_bagging_dataloader(dataset, batch_size, subsample_size):
        indices = torch.randint(0, len(dataset), (subsample_size,))
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return dataloader
    
    train_dataloaders = [create_bagging_dataloader(train_dataset, constants.training_batch_size, int(len(train_dataset)*0.3333)) for _ in range(num_members)]

    model = UNetWithAttentionMini(1, 1, output_shape=(64,64)).to(constants.device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()


    for epoch in range(continue_epoch or 0, max_epoch):
        for member_idx in range(num_members):

            train_dataloader = train_dataloaders[member_idx]

            if epoch == 0:
                print('New model')
                model = UNetWithAttentionMini(1, 1, output_shape=(64,64)).to(constants.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            else:
                model = UNetWithAttentionMini(1, 1, output_shape=(64,64)).to(constants.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                checkpoint = torch.load(f'{constants.checkpoints_dir}ens_{domain}_{member_idx}.pt')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('loaded weights')

            def train(domain, model, dataloader, criterion, optimizer, device, pad=False, plot=False):
                model.train()
                losses = []

                if plot:
                    lats, lons, input_lats, input_lons = get_lats_lons(domain, pad)
                    random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
                    plotted = 0

                for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

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


            train_loss = train(domain, model, train_dataloader, criterion, optimizer, constants.device, pad=pad, plot=False)

            if os.path.exists(f'{constants.checkpoints_dir}ens_{domain}_{member_idx}.pt'):
                os.remove(f'{constants.checkpoints_dir}ens_{domain}_{member_idx}.pt')

            #save model, optimizer, epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, f'{constants.checkpoints_dir}ens_{domain}_{member_idx}.pt')


            torch.cuda.empty_cache()
            import gc
            gc.collect()

        def test_ens(domain, models, dataloader, criterion, device, pad=False, plot=True):
            losses = []
            bilinear_losses = []

            if plot:
                lats, lons, input_lats, input_lons = get_lats_lons(domain, pad)
                random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
                plotted = 0

            for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
                inputs, targets = inputs.to(device), targets.to(device)

                ens_outputs = []
                for model in models:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(inputs)
                    ens_outputs.append(outputs)

                outputs = torch.stack(ens_outputs).mean(0)

                loss = criterion(outputs, targets)
                
                cropped_inputs = inputs[:, 1:-1, 1:-1]
                interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
                
                bilinear_loss = criterion(interpolated_inputs, targets)

                losses.append(loss.item())
                bilinear_losses.append(bilinear_loss.item())

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

            return np.mean(losses), np.mean(bilinear_losses)
        
        models = [UNetWithAttentionMini(1, 1, output_shape=(64,64)).to(constants.device) for _ in range(num_members)]
        for m in range(num_members):
            checkpoint = torch.load(f'{constants.checkpoints_dir}ens_{domain}_{m}.pt')
            models[m].load_state_dict(checkpoint['model_state_dict'])
            models[m].eval()

        test_loss, bilinear_loss = test_ens(domain, models, test_dataloader, criterion, constants.device, pad=pad, plot=True)
        print(f'Epoch {epoch} - Test Loss: {test_loss} - Bilinear Loss: {bilinear_loss}')

        del models
        torch.cuda.empty_cache()
        import gc
        gc.collect()