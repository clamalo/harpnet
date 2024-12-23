import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.model import UNetWithAttention
from src.constants import TORCH_DEVICE, CHECKPOINTS_DIR

def train_test(tile, train_dataloader, test_dataloader, start_epoch=0, end_epoch=20):

    epochs = list(range(start_epoch, end_epoch))
    torch.manual_seed(42)

    model = UNetWithAttention().to(TORCH_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    if start_epoch != 0:
        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, str(tile), f'{start_epoch-1}_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in epochs:
        train_losses = []
        for i, (inputs, targets, times) in enumerate(train_dataloader):
            inputs, targets = inputs.to(TORCH_DEVICE), targets.to(TORCH_DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        bilinear_test_losses = []
        for i, (inputs, targets, times) in enumerate(test_dataloader):
            inputs, targets = inputs.to(TORCH_DEVICE), targets.to(TORCH_DEVICE)
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, targets)

            cropped_inputs = inputs[:, 1:-1, 1:-1]
            interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
            bilinear_loss = criterion(interpolated_inputs, targets)

            test_losses.append(loss.item())
            bilinear_test_losses.append(bilinear_loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        test_loss = sum(test_losses) / len(test_losses)
        bilinear_test_loss = sum(bilinear_test_losses) / len(bilinear_test_losses)

        print(f'Epoch {epoch}: Train loss = {train_loss}, Test loss = {test_loss}, Bilinear test loss = {bilinear_test_loss}')

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_test_loss': bilinear_test_loss
        }, os.path.join(CHECKPOINTS_DIR, f'{tile}/{epoch}_model.pt'))