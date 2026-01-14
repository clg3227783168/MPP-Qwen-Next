from torch.utils.data import Dataset
from torch.utils.dataloader import default_collate

class BaseDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.collater = default_collate

    @classmethod
    def from_config(cls, config):
        return cls(**config)