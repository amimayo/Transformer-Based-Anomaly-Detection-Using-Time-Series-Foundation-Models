import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class SMDDataset(Dataset):

    def __init__(self, data, window_size, is_channel_independent):

        self.data = data
        self.window_size = window_size
        self.is_channel_independent = is_channel_independent

    def __len__(self):

        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):

        window = self.data[idx : idx + self.window_size]
        window = torch.tensor(window, dtype=torch.float32)

        if self.is_channel_independent:

            window = window.transpose(0,1)

            return window

        return window
    
def get_dataloaders(train_data, test_data, window_size, num_workers, pin_memory, batch_size, is_channel_independent):

    train_dataset = SMDDataset(train_data, window_size, is_channel_independent)
    test_dataset = SMDDataset(test_data, window_size, is_channel_independent)

    train_dataloader = DataLoader(train_dataset, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_dataloader, test_dataloader