import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
from utils.model import UNetWithAttention, Discriminator
from utils.utils2 import *
import matplotlib.pyplot as plt
import cartopy
from metpy.plots import USCOUNTIES
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")
import random
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


# domain = 14
domain = 32
LOAD = False
first_month = (1979, 10)
last_month = (1981, 9)
train_test = 0.2
continue_epoch = None
max_epoch = 10


if LOAD:
    setup(domain)
    create_grid_domains()
    xr_to_np(domain, first_month, last_month)

train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths = create_paths(domain, first_month, last_month, train_test)

train_dataset, train_dataloader = create_dataloader(train_input_file_paths, train_target_file_paths, shuffle=True)
test_dataset, test_dataloader = create_dataloader(test_input_file_paths, test_target_file_paths, shuffle=False)

print(len(train_dataloader), next(iter(train_dataloader))[0].numpy().shape, next(iter(train_dataloader))[1].numpy().shape)
print(len(test_dataloader), next(iter(test_dataloader))[0].numpy().shape, next(iter(test_dataloader))[1].numpy().shape)

generator = UNetWithAttention(1, 1, output_shape=(32, 32)).to('mps')
discriminator = Discriminator(in_channels=1).to('mps')
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
criterion_GAN = nn.BCELoss()
criterion_content = nn.MSELoss()


for epoch in range(continue_epoch or 0, max_epoch):
    G_loss, D_loss = train_gan(domain, generator, discriminator, train_dataloader, criterion_GAN, criterion_content, optimizer_G, optimizer_D, 'mps', plot=True)
    test_loss = test_gan(domain, generator, test_dataloader, criterion_content, 'mps')
    print(f'Epoch {epoch} - G Loss: {G_loss:.4f} - D Loss: {D_loss:.4f} - Test Loss: {test_loss:.4f}')
    # checkpoint = {
    #     'generator_state_dict': generator.state_dict(),
    #     'discriminator_state_dict': discriminator.state_dict(),
    #     'optimizer_G_state_dict': optimizer_G.state_dict(),
    #     'optimizer_D_state_dict': optimizer_D.state_dict(),
    #     'G_loss': G_loss,
    #     'D_loss': D_loss,
    #     'test_loss': test_loss
    # }
    # torch.save(checkpoint, f'checkpoints/{domain}/{epoch}_model.pt')