from config import *
from setup import setup
from data_preprocessing import preprocess_data
from generate_dataloaders import generate_dataloaders

setup()
preprocess_data()
train_loader, test_loader = generate_dataloaders()

print("DataLoaders created successfully!")