from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test

# Variables
domain = 15
start_month = (1979, 10)
end_month = (1980, 9)
train_test_ratio = 0.2


setup(domain)

xr_to_np(domain, start_month, end_month)

train_dataloader, test_dataloader = generate_dataloaders(domain, start_month, end_month, train_test_ratio)

train_test(domain, train_dataloader, test_dataloader, epochs=20)