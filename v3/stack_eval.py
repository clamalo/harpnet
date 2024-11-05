from src.generate_dataloaders import generate_dataloaders

start_month = (1979, 10)
end_month = (1979, 10)
train_test_ratio = 0.2

train_dataloader, test_dataloader = generate_dataloaders(start_month, end_month, train_test_ratio)

print(len(train_dataloader), len(test_dataloader))