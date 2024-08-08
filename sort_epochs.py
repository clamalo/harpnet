import torch
import os
from tqdm import tqdm


def sort_epochs():
    if not os.path.exists('checkpoints/best/'):
        os.makedirs('checkpoints/best/')

    patches = [f for f in os.listdir('checkpoints/') if f != 'best' and f != '.DS_Store']

    for patch in patches:
        checkpoint_dir = f'checkpoints/{patch}/'

        # Get all checkpoint files in the directory
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_model.pt')]

        if len(checkpoint_files) == 0:
            continue

        best_members = []

        for checkpoint_file in tqdm(checkpoint_files):
            # Extract the epoch number from the filename
            epoch = int(checkpoint_file.split('_')[0])
            
            checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
            train_loss = checkpoint['train_loss']
            test_loss = checkpoint['test_loss']
            best_members.append((epoch, train_loss, test_loss))

        # Sort by test_loss (ascending) and select the top 5
        best_members = sorted(best_members, key=lambda x: x[2])[:5]

        best_member = best_members[0]
        best_checkpoint = torch.load(os.path.join(checkpoint_dir, f'{best_member[0]}_model.pt'))
        torch.save(best_checkpoint, os.path.join('checkpoints/best/', f'{patch}_model.pt'))

if __name__ == '__main__':
    sort_epochs()