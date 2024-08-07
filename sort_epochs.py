import torch
from tqdm import tqdm

patch = 15

best_members = []

for epoch in tqdm(range(31)):
    checkpoint = torch.load(f'checkpoints/{patch}/{epoch}_model.pt')
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    best_members.append((epoch, train_loss, test_loss))

# Sort by test_loss (ascending) and select the top 5
best_members = sorted(best_members, key=lambda x: x[2])[:5]

# Print the results
for member in best_members:
    print(f'Epoch: {member[0]}, Train Loss: {member[1]}, Test Loss: {member[2]}')