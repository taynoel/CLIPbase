from abc import ABCMeta, abstractmethod
from typing import Dict
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class AbstractDFDataset(Dataset ,metaclass=ABCMeta):
    def __init__(self, returnWithLabel=True, doAugment=False):
        self.returnWithLabel = returnWithLabel
        self.doAugment = doAugment
        self._data_list: DataFrame = None

    @property
    def data_list(self):
        return self._data_list
    
    @data_list.setter
    def data_list(self, df: DataFrame):
        self._data_list = df

    def __len__(self):
        return len(self._data_list)
    

    def __getitem__(self, idx: int):
        return self.get_input_and_label(idx) if self.returnWithLabel else self.get_input_only(idx)


    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, pin_memory=False) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn)
    
    @abstractmethod
    def get_input_and_label(self, idx: int) -> dict:
        pass
    
    @abstractmethod
    def get_input_only(self, idx: int) -> dict:
        pass

    @abstractmethod
    def collate_fn(self, batch) -> Dict[str, Tensor]:
        """ Used as input for collate_fn in Dataloader.
        By default, it is assumed that stacking of samples from the batch is performed here.
        Also, need to make sure relevant data is converted to Tensor
        Depending on returnWithLabel, batch may come from get_input_only or get_input_and_label,
        so need to make sure the code can handle the possible differences of both
        """
        pass

    