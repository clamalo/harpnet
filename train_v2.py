import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
from utils.model import UNetWithAttention
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

model = UNetWithAttention(1, 1, output_shape=(32,32)).to('mps')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

if continue_epoch:
    checkpoint = torch.load(f'checkpoints/{domain}/{continue_epoch}_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(continue_epoch or 0, max_epoch):
    train_loss = train(domain, model, train_dataloader, criterion, optimizer, 'mps', plot=True)
    test_loss = test(domain, model, test_dataloader, criterion, 'mps')
    print(f'Epoch {epoch} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    torch.save(checkpoint, f'checkpoints/{domain}/{epoch}_model.pt')