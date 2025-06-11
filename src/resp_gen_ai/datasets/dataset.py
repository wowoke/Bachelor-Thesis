"""Base Dataset class for resp_gen_ai"""
from torch.utils.data import Dataset


class RespGenAIDataset(Dataset):
    """Base Dataset class for resp_gen_ai"""

    def __init__(self):
        super().__init__()

    def load_train_data(self):
        raise NotImplementedError

    def load_val_data(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
