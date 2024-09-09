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
from utils.model import UNetWithAttention
from utils.utils import *
import utils.constants as constants


domains = [17]


setup()
for domain in domains:
    LOAD = True
    first_month = (1979, 10)
    last_month = (1980, 9)
    train_test = 0.2
    continue_epoch = False
    max_epoch = 1
    pad = True


    if LOAD:
        setup(domain)
        create_grid_domains()
        xr_to_np(domain, first_month, last_month, pad=pad)

    train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths = create_paths(domain, first_month, last_month, train_test)

    train_dataset = create_dataset(train_input_file_paths, train_target_file_paths)
    test_dataset, test_dataloader = create_dataloader(test_input_file_paths, test_target_file_paths, shuffle=False)

    from torch.utils.data import DataLoader, SubsetRandomSampler
    def create_bagging_dataloader(dataset, batch_size, subsample_size):
        indices = torch.randint(0, len(dataset), (subsample_size,))
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return dataloader
    
    train_dataloaders = [create_bagging_dataloader(train_dataset, constants.training_batch_size, int(len(train_dataset)*0.5)) for _ in range(3)]


    models = [UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.device) for _ in range(3)]
    print(f'Number of parameters: {sum(p.numel() for p in models[0].parameters())}')
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4) for model in models]
    criterion = nn.MSELoss()


    for epoch in range(continue_epoch or 0, max_epoch):
        for model, optimizer, train_dataloader in zip(models, optimizers, train_dataloaders):
            train_loss = train(domain, model, train_dataloader, criterion, optimizer, constants.device, pad=pad, plot=False)

        model = models[0]  # Initialize with the first model
        for model_i in models[1:]:
            for param, param_i in zip(model.parameters(), model_i.parameters()):
                param.data += param_i.data  # Sum parameters
        for param in model.parameters():
            param.data /= len(models)  # Average parameters


        test_loss, bilinear_loss = test(domain, model, test_dataloader, criterion, constants.device, pad=pad, plot=True)
        print(f'Epoch {epoch} - Test Loss: {test_loss} - Bilinear Loss: {bilinear_loss}')