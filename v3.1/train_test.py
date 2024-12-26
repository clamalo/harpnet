from config import (
    TRAINING_PROGRESS_BAR,
    NUM_EPOCHS,
    CHECKPOINTS_DIR,
    MODEL_NAME,
    DEVICE
)
import importlib
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


def train(train_dataloader, model, optimizer, criterion, device):
    """
    Train for one epoch. Data shape:
      'input':  (B, 2, fLat, fLon)
      'target': (B, fLat, fLon)
    """
    model.train()
    running_loss = 0.0
    total_steps = 0

    data_iter = tqdm(train_dataloader, desc="Training epoch") if TRAINING_PROGRESS_BAR else train_dataloader

    for batch in data_iter:
        # (B, 2, fLat, fLon)
        inputs = batch['input'].float().to(device)
        # (B, fLat, fLon)
        targets = batch['target'].float().to(device)
        # Model expects (B, 2, fLat, fLon) -> (B, 1, fLat, fLon) output
        # No need to interpolate, already sized to (fLat, fLon).

        optimizer.zero_grad()
        outputs = model(inputs)  # (B, 1, fLat, fLon)
        # Reshape targets to (B, 1, fLat, fLon) so we can do MSE easily
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_steps += 1

    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
    return avg_loss


def test(test_dataloader, model, criterion, device):
    """
    Evaluate for one epoch. Data shape is the same as train().
    """
    model.eval()
    running_loss = 0.0
    total_steps = 0

    data_iter = tqdm(test_dataloader, desc="Testing epoch") if TRAINING_PROGRESS_BAR else test_dataloader

    with torch.no_grad():
        for batch in data_iter:
            inputs = batch['input'].float().to(device)    # (B, 2, fLat, fLon)
            targets = batch['target'].float().to(device)  # (B, fLat, fLon)

            outputs = model(inputs)                       # (B, 1, fLat, fLon)
            loss = criterion(outputs, targets.unsqueeze(1))
            running_loss += loss.item()
            total_steps += 1

    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
    return avg_loss


def train_test(train_dataloader, test_dataloader):
    """
    Full train/test loop for NUM_EPOCHS.
    """
    # Decide on device
    if DEVICE.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif DEVICE.lower() == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Dynamically load the model
    model_module = importlib.import_module(f"{MODEL_NAME}")
    ModelClass = model_module.Model
    model = ModelClass().to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Run training & testing loops
    for epoch in range(0, NUM_EPOCHS):
        train_loss = train(train_dataloader, model, optimizer, criterion, device)
        test_loss = test(test_dataloader, model, criterion, device)

        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"epoch_{epoch}_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }, checkpoint_path)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    print("Training complete!")