from src.generate_dataloaders import generate_dataloaders
from src.model import UNetWithAttention
from src.tile_weight_mask import tile_weight_mask
import torch
from src.constants import TORCH_DEVICE
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



array = tile_weight_mask()

model = UNetWithAttention(1, 1, output_shape=(64,64)).to(TORCH_DEVICE)

start_month = (1979, 10)
end_month = (1980, 9)
train_test_ratio = 0.2



def inference(start_month, end_month, train_test_ratio, tile, model):
    checkpoint = torch.load(f'v3_checkpoints/best/{tile}_model.pt', map_location=torch.device(TORCH_DEVICE))
    if tile == 4:
        print(checkpoint['test_loss'], checkpoint['bilinear_test_loss'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    _, test_dataloader = generate_dataloaders(tile, start_month, end_month, train_test_ratio)
    model.eval()
    outputs = []
    for i, (inputs, targets, times) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        inputs, targets = inputs.to(TORCH_DEVICE), targets.to(TORCH_DEVICE)
        with torch.no_grad():
            output = model(inputs).cpu().detach().numpy()
        batch_array = np.tile(array, (output.shape[0], 1, 1))
        output = output * batch_array
        # output = batch_array
        outputs.append(output)
    outputs = np.concatenate(outputs, axis=0)

    return outputs





outputs = {}

for tile in range(0,5):
    tile_outputs = inference(start_month, end_month, train_test_ratio, tile, model)
    outputs[tile] = tile_outputs

bottom_two = np.concatenate([outputs[0], outputs[1]], axis=2)
top_two = np.concatenate([outputs[2], outputs[3]], axis=2)
full = np.concatenate([bottom_two, top_two], axis=1)

# full[:, 32:96, 32:96] = full[:, 32:96, 32:96]+outputs[4]


def criterion(outputs, targets):
    return np.mean(np.square(outputs - targets))


losses = []
_, test_dataloader = generate_dataloaders(4, start_month, end_month, train_test_ratio)
for i, (inputs, targets, times) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

    targets = targets.cpu().detach().numpy()

    full_start = i*32
    full_end = min((i+1)*32, full.shape[0])

    loss = criterion(targets, full[full_start:full_end, 32:96, 32:96])

    losses.append(loss)

print(np.mean(losses))




print(full.shape)

sum = np.sum(full, axis=0)

cf = plt.imshow(np.flip(sum, axis=0), vmin=0, vmax=350)
plt.colorbar(cf)
plt.show()