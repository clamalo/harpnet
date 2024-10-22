import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.model import UNetWithAttention
import src.constants as constants

def train_test(tile, train_dataloader, test_dataloader, epochs=20):

    torch.manual_seed(42)

    model = UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        # train
        train_losses = []
        for i, (inputs, targets, times) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, targets = inputs.to(constants.torch_device), targets.to(constants.torch_device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # # encourage physical constraint:
            # coarsened_outputs = outputs.view(inputs.shape[0], 8, 8, 8, 8).mean(dim=(3, 4))
            # coarsen_loss = criterion(coarsened_outputs, inputs[:, 1:-1, 1:-1])
            # loss = loss + 0.1*coarsen_loss

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())



        # test
        model.eval()
        test_losses = []
        bilinear_test_losses = []
        for i, (inputs, targets, times) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            inputs, targets = inputs.to(constants.torch_device), targets.to(constants.torch_device)
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

        # save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_test_loss': bilinear_test_loss
        }, os.path.join(constants.checkpoints_dir, f'{tile}/{epoch}_model.pt'))
