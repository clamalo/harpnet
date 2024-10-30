from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from src.constants import PROCESSED_DIR

def generate_dataloaders(first_month, last_month, train_test_ratio):

    class MemMapDataset(Dataset):
        def __init__(self, inputs, targets, times):
            self.inputs = inputs
            self.targets = targets
            self.times = times.astype('float64')
        def __len__(self):
            return self.inputs.shape[0]
        def __getitem__(self, idx):
            # Assuming times is your datetime array
            return self.inputs[idx], self.targets[idx], self.times[idx]

    # Generate list of months between first_month and last_month inclusive
    first = datetime(*first_month, day=1)
    last = datetime(*last_month, day=1)
    months = []
    current = first
    while current <= last:
        months.append(current)
        current += relativedelta(months=1)

    # Create file paths using list comprehensions
    input_file_paths = [
        os.path.join(PROCESSED_DIR, f'input_{m.year}_{m.month:02d}.npy') for m in months
    ]
    target_file_paths = [
        os.path.join(PROCESSED_DIR, f'target_{m.year}_{m.month:02d}.npy') for m in months
    ]
    times_file_paths = [
        os.path.join(PROCESSED_DIR, f'times_{m.year}_{m.month:02d}.npy') for m in months
    ]

    # Load and concatenate arrays
    input_arr = np.concatenate([np.load(fp) for fp in input_file_paths])
    target_arr = np.concatenate([np.load(fp) for fp in target_file_paths])
    times_arr = np.concatenate([np.load(fp) for fp in times_file_paths])

    # Shuffle the data
    np.random.seed(42)
    indices = np.random.permutation(len(times_arr))
    input_arr, target_arr, times_arr = input_arr[indices], target_arr[indices], times_arr[indices]

    print(times_arr[:3].astype('datetime64[s]'))

    # Split the data
    split_idx = int(train_test_ratio * len(input_arr))
    train_input, test_input = input_arr[split_idx:], input_arr[:split_idx]
    train_target, test_target = target_arr[split_idx:], target_arr[:split_idx]
    train_times, test_times = times_arr[split_idx:], times_arr[:split_idx]

    # Create datasets
    train_dataset = MemMapDataset(train_input, train_target, train_times)
    test_dataset = MemMapDataset(test_input, test_target, test_times)

    # Create DataLoaders with deterministic shuffling for training
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader