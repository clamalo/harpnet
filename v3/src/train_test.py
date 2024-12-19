import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.model import UNetWithAttention
from src.constants import TORCH_DEVICE, CHECKPOINTS_DIR

def train_test(train_dataloader, test_dataloader, start_epoch=0, end_epoch=20, focus_tile=None):
    torch.manual_seed(42)
    model = UNetWithAttention().to(TORCH_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    if start_epoch != 0:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{start_epoch-1}_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=TORCH_DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_losses = []
        train_iter = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]", unit="batch")

        for batch in train_iter:
            inputs, elev_data, targets, times, tile_ids = batch
            
            # Interpolate inputs to 64x64
            inputs = torch.nn.functional.interpolate(inputs, size=(64, 64), mode='nearest')
            
            # Ensure elev_data is also (N,1,64,64). If not, interpolate as well:
            elev_data = torch.nn.functional.interpolate(elev_data, size=(64,64), mode='nearest')
            
            # Concatenate inputs and elevation along the channel dimension
            # Now inputs has shape (N,1,64,64) and elev_data has shape (N,1,64,64),
            # so after concatenation: (N,2,64,64)
            inputs = torch.cat([inputs, elev_data], dim=1)
            
            inputs = inputs.to(TORCH_DEVICE)
            targets = targets.to(TORCH_DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)  # model now expects (N,C,64,64), C>=2 if elevation is included
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_iter.set_postfix(loss=f"{loss.item():.4f}")


        model.eval()
        test_losses = []
        bilinear_test_losses = []
        focus_tile_losses = []

        test_iter = tqdm(test_dataloader, desc=f"Epoch {epoch} [Test]", unit="batch")
        with torch.no_grad():
            for batch in test_iter:
                inputs, elev_data, targets, times, tile_ids = batch

                # Interpolate inputs and elevation to 64x64
                inputs = torch.nn.functional.interpolate(inputs, size=(64, 64), mode='nearest')
                elev_data = torch.nn.functional.interpolate(elev_data, size=(64, 64), mode='nearest')
                
                # Concatenate
                inputs = torch.cat([inputs, elev_data], dim=1)
                
                inputs = inputs.to(TORCH_DEVICE)
                targets = targets.to(TORCH_DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                cropped_inputs = inputs[:,0:1,1:-1,1:-1]
                interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs, size=(64, 64), mode='bilinear')
                bilinear_loss = criterion(interpolated_inputs, targets)

                test_losses.append(loss.item())
                bilinear_test_losses.append(bilinear_loss.item())
                test_iter.set_postfix(loss=f"{loss.item():.4f}", bilinear_loss=f"{bilinear_loss.item():.4f}")

                # If focus_tile is specified, compute tile-specific test loss
                if focus_tile is not None:
                    mask = (tile_ids == focus_tile)
                    if mask.any():
                        focus_outputs = outputs[mask]
                        focus_targets = targets[mask]
                        focus_loss = criterion(focus_outputs, focus_targets)
                        focus_tile_losses.append(focus_loss.item())

        train_loss = sum(train_losses) / len(train_losses) if train_losses else float('inf')
        test_loss = sum(test_losses) / len(test_losses) if test_losses else float('inf')
        bilinear_test_loss = sum(bilinear_test_losses) / len(bilinear_test_losses) if bilinear_test_losses else float('inf')

        print(f'Epoch {epoch}: Train loss = {train_loss}, Test loss = {test_loss}, Bilinear test loss = {bilinear_test_loss}')

        if focus_tile is not None:
            if focus_tile_losses:
                focus_tile_test_loss = sum(focus_tile_losses) / len(focus_tile_losses)
                print(f"Epoch {epoch}: Test loss for tile {focus_tile} = {focus_tile_test_loss}")
            else:
                print(f"Epoch {epoch}: No samples found for tile {focus_tile} in the test set.")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_test_loss': bilinear_test_loss
        }, os.path.join(CHECKPOINTS_DIR, f'{epoch}_model.pt'))