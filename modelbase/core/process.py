from typing import Type
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from torch import save, cuda, Tensor, no_grad
from torch.optim import Optimizer
from torch.nn.parallel.data_parallel import DataParallel
from .dataset import AbstractDFDataset, DataLoader
from .network import AbstractNetwork
from .datastruct import IterData, EpochData


class AvgMeter:
    def __init__(self):
        self.count = self.sum = self.avg = 0

    def update(self, value: float):
        self.count += 1
        self.sum += value
        self.avg = self.sum / self.count


class DTScheduler:
    def step(self): ...


class AbstractProcess(metaclass=ABCMeta):
    def __init__(self):
        self.train_dataset: AbstractDFDataset = None
        self.train_dataloader: DataLoader = None
        self.validation_dataset: AbstractDFDataset = None
        self.validation_dataloader: DataLoader = None
        self.network: AbstractNetwork = None
        self.optimizer: Optimizer = None
        self.optim_scheduler: DTScheduler = None

        self.device: str = 'cuda'
        self.doDataParallel = False

        self.num_epochs: int = None
        self.network_class: Type[AbstractNetwork] = None

    # ==========================================================
    # Abstract methods
    # ==========================================================
    @abstractmethod
    def prepare_dataloader(self):
        """ Create dataloaders
        Assign to .train_dataloader, .validation_dataloader ...
        """
        pass

    @abstractmethod
    def prepare_network(self):
        """ Network definition and initialization
        Assign to .network
        Initialize with:
        if .pretrained_model_file:
            .load_model(self.pretrained_model_file)
        """
        pass

    @abstractmethod
    def prepare_optimizer(self):
        """ Create optimizer and scheduler
        Assign to .optimizer and .optim_scheduler (optional)
        """
        pass

    @abstractmethod
    def feedforward(self, proc_data: IterData) -> IterData:
        """ Forward input to network and get output
        By default, use IterData.data as input, and save output to IterData.output
        """
        pass

    @abstractmethod
    def feedforward_loss(self, proc_data: IterData) -> IterData:
        """ Get loss after forward
        By default, use IterData.output as input, and save loss to IterData.loss_for_iter
        IterData.loss_for_iter is used for backpropagation
        """
        pass

    # ==========================================================
    # Defined methods
    # ==========================================================
    def prepare_parallel(self):
        if self.doDataParallel and cuda.device_count() > 1:
            self.network = DataParallel(self.network)

    def load_model(self, file_name: str):
        self.network.load_pretrained_param_from_file(file_name)

    def save_model(self, file_name: str):
        _state_dict = self.network.module.state_dict() if hasattr(self.network, 'module')\
            else self.network.state_dict()
        save(_state_dict, file_name)

    def loaded_to_process_data(self, proc_data: IterData, loaded_data: dict) -> IterData:
        def convert_device(item, device):
            if type(item) is Tensor and device == 'cuda':
                return item.cuda()
            return item
        proc_data.data = {key: convert_device(item, self.device) for key, item in loaded_data.items()}
        return proc_data

    def backprop_update(self, proc_data: IterData) -> None:
        self.optimizer.zero_grad()
        proc_data.loss_for_iter.backward()
        self.optimizer.step()

    # ==========================================================
    # Template/Example methods
    # ==========================================================
    def prepare(self):
        self.prepare_dataloader()
        self.prepare_network()
        self.prepare_optimizer()
        self.prepare_parallel()

    def run_epoch(self, dataloader: DataLoader, updateParam=False) -> EpochData:
        loss_meter = AvgMeter()
        tqdm_obj = tqdm(dataloader)
        for _, f_data_dic in enumerate(tqdm_obj):
            proc_data = IterData()
            self.loaded_to_process_data(proc_data, f_data_dic)
            self.feedforward(proc_data)
            self.feedforward_loss(proc_data)
            if updateParam:
                self.backprop_update(proc_data)
            loss_meter.update(proc_data.loss_for_iter.item())
            tqdm_obj.set_postfix(loss=loss_meter.avg)
        return EpochData(loss_for_epoch=loss_meter.avg)

    def train(self):
        self.network.cuda()
        best_eval_loss = float('inf')
        for _ in range(self.num_epochs):
            self.network.train()
            self.run_epoch(self.train_dataloader, updateParam=True)

            with no_grad():
                self.network.eval()
                epoch_data = self.run_epoch(self.validation_dataloader, updateParam=False)
            if self.optim_scheduler:
                self.optim_scheduler.step(epoch_data.loss_for_epoch)
            if epoch_data.loss_for_epoch < best_eval_loss:
                best_eval_loss = epoch_data.loss_for_epoch
                self.save_model('best.pt')
                print("Saved Best Model!")
