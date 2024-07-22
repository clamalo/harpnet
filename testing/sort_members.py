import torch
import os
import re
from tqdm import tqdm

def load_checkpoints(directory, sort_by='train_loss', start_epoch=0, end_epoch=5):
    checkpoint_data = []
    
    # Load all checkpoint files from the specified directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.pt'):  # Ensure it's a PyTorch checkpoint file
            # Extract epoch number from the filename using a regular expression
            match = re.search(r'e(\d+)\.pt$', filename)
            if match:
                epoch = int(match.group(1))  # Convert extracted epoch number to integer
                # Filter checkpoints to only include those within the specified epoch range
                if start_epoch <= epoch <= end_epoch:
                    filepath = os.path.join(directory, filename)
                    checkpoint = torch.load(filepath)
                    
                    # Get the loss value for the specified sort_by criterion
                    loss = checkpoint.get(sort_by, float('inf'))  # Use float('inf') if loss key does not exist
                    train_loss = checkpoint.get('train_loss', float('inf'))
                    
                    # Append a tuple of (epoch, loss) to the list
                    checkpoint_data.append((epoch, loss, train_loss))
    
    # Sort the list by the specified loss (ascending order)
    checkpoint_data.sort(key=lambda x: x[1])
    
    # Print the sorted epochs based on the specified loss
    for data in checkpoint_data[:5]:
        print(f'Epoch: {data[0]}, {sort_by}: {data[1]}, Train Loss: {data[2]}')

# Example usage
directory = '/Users/clamalo/documents/harpnet/8km_checkpoints/'
load_checkpoints(directory, sort_by='test_loss', start_epoch=0, end_epoch=50)