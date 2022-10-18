import os
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from utils.augmentations import Jittering


class SensorDataset(Dataset):
    def __init__(self, data_path, transforms=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.files = [os.path.join(data_path, dir_) for dir_ in os.listdir(data_path)]
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signals = np.load(self.files[idx])
        label_text = os.path.basename(self.files[idx]).split('.')[0].split('_')[2]
        label = 1 if label_text == 'cycle' else 0
        if self.transforms is not None:
            signals = self.transforms(signals)
        return signals, label


class SensorDataModule(LightningDataModule):
    def __init__(self,
            train_path,
            val_path,
            test_path,
            batch_size,
            train_transforms = {},
            test_transforms = {},
            num_workers = 1):
        super().__init__()
        # paths
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        # batch and transforms
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        # other
        self.num_workers = num_workers

        self._init_dataloaders()
        self.save_hyperparameters("batch_size")

    def _init_dataloaders(self):
        train_dataset = self._create_train_dataset()
        test_dataset = self._create_test_dataset()
        val_dataset = self._create_val_dataset()

        self.train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

        if len(os.listdir(self.val_path)) == 0:
            self.val = None
        else:
            self.val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

    def _create_train_dataset(self):
        return SensorDataset(
            data_path=self.train_path,
            transforms=self.train_transforms
        )
        
    def _create_val_dataset(self):
        return SensorDataset(
            data_path=self.val_path,
            transforms=self.test_transforms
        )

    def _create_test_dataset(self):
        return SensorDataset(
            data_path=self.test_path,
            transforms=self.test_transforms
        )

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test