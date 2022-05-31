from __future__ import annotations
from typing import Type
import torch
import torch.nn.functional as F
from modelbase.core import IterData, EpochData, AbstractNetwork, AbstractProcess
from modelbase.core import utils
from modelbase.network_store.clip_net import ClipNet
from modelbase.dataset_store.flickr8kcaption_dataset import DatasetBuilder


class _DTBuilderParam:
    name: str

    class info:
        process_module: str
        network_module: str
        dataset_module: str

    class parameters:
        num_workers: int
        num_epochs: int
        image_encoder_lr: float
        text_encoder_lr: float
        projection_lr: float
        weight_decay: float
        temperature: float
        scheduler_patience: int
        scheduler_factor: float
        batch_size: int
        pretrained_model_file: str
        data_csv_file: str
        image_folder: str


class ProcessBuilder:
    def __init__(self): ...

    def build_from_yaml(self, yaml_file: str) -> Process:
        params: _DTBuilderParam = utils.read_param_yaml(yaml_file)
        process = Process()
        _prm = params.parameters
        [process.num_workers, process.num_epochs, process.temperature, process.image_encoder_lr,
         process.text_encoder_lr, process.projection_lr, process.weight_decay, process.scheduler_patience,
         process.scheduler_factor, process.batch_size, process.pretrained_model_file,
         process.csv_file, process.image_folder] =\
            [_prm.num_workers, _prm.num_epochs, _prm.temperature, _prm.image_encoder_lr, _prm.text_encoder_lr,
             _prm.projection_lr, _prm.weight_decay, _prm.scheduler_patience, _prm.scheduler_factor,
             _prm.batch_size, _prm.pretrained_model_file, _prm.data_csv_file, _prm.image_folder]
        process.network_class = ClipNet
        process.prepare()
        return process


class Process(AbstractProcess):
    network: ClipNet

    def __init__(self):
        super().__init__()
        self.num_epochs: int = None
        self.network_class: Type[AbstractNetwork] = None

        self.image_encoder_lr: float = None
        self.text_encoder_lr: float = None
        self.projection_lr: float = None
        self.weight_decay: float = None
        self.batch_size: int = None
        self.num_workers: int = None
        self.temperature: float = None
        self.scheduler_patience: int = None
        self.scheduler_factor: float = None

        self.pretrained_model_file: str = None
        self.csv_file: str = None
        self.image_folder: str = None

    def prepare_dataloader(self):
        dataset_builder = DatasetBuilder(self.batch_size, self.num_workers)
        dataset_builder.build_train_validation_dataset(self.csv_file, self.image_folder)
        self.train_dataset = dataset_builder.train_dataset
        self.validation_dataset = dataset_builder.validation_dataset
        self.train_dataloader = dataset_builder.train_dataloader
        self.validation_dataloader = dataset_builder.validation_dataloader

    def prepare_network(self):
        self.network = self.network_class()
        if self.pretrained_model_file:
            self.load_model(self.pretrained_model_file)

    def prepare_optimizer(self):
        params = [
            {"params": self.network.image_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.network.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {"params": self.network.image_projection.parameters(),
             "lr": self.projection_lr, "weight_decay": self.weight_decay},
            {"params": self.network.text_projection.parameters(),
             "lr": self.projection_lr, "weight_decay": self.weight_decay}]
        self.optimizer = torch.optim.AdamW(params, weight_decay=0.)
        if self.scheduler_factor is not None and self.scheduler_patience is not None:
            self.optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=self.scheduler_patience, factor=self.scheduler_factor)

    def _get_loss(self, img_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        logits = (text_embedding @ img_embedding.T) / self.temperature
        images_similarity = img_embedding @ img_embedding.T
        texts_similarity = text_embedding @ text_embedding.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        return self._cross_entropy(logits, targets)

    def _cross_entropy(self, preds, targets) -> torch.Tensor:
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        return loss

    def feedforward(self, proc_data: IterData) -> IterData:
        img_embedding, text_embedding = self.network(proc_data.data)
        proc_data.output = {'img_embedding': img_embedding, 'text_embedding': text_embedding}
        return proc_data

    def feedforward_loss(self, proc_data: IterData) -> IterData:
        img_emd, text_emd = proc_data.output['img_embedding'], proc_data.output['text_embedding']
        loss_tensor = self._get_loss(img_emd, text_emd)
        proc_data.loss_for_iter = loss_tensor.mean()
        return proc_data

    def run_epoch(self, *args, **kwargs) -> EpochData:
        return super().run_epoch(*args, **kwargs)

    def train(self):
        self.network.cuda()
        best_eval_loss = float('inf')
        for _ in range(self.num_epochs):
            self.network.train()
            self.run_epoch(self.train_dataloader, updateParam=True)

            with torch.no_grad():
                self.network.eval()
                epoch_data = self.run_epoch(self.validation_dataloader, updateParam=False)
            if self.optim_scheduler:
                self.optim_scheduler.step(epoch_data.loss_for_epoch)
            if epoch_data.loss_for_epoch < best_eval_loss:
                best_eval_loss = epoch_data.loss_for_epoch
                self.save_model('best.pt')
                print("Saved Best Model!")


def example():
    process = Process()
    process.num_epochs = 4
    process.network_class = ClipNet
    process.image_encoder_lr = 1e-4
    process.text_encoder_lr = 1e-5
    process.projection_lr = 1e-3
    process.weight_decay = 1e-3
    process.batch_size = 32
    process.num_workers = 4
    process.temperature = 1.0
    process.scheduler_patience = 1
    process.scheduler_factor = 0.4
    process.csv_file = '../flickr8k/captions.txt'
    process.image_folder = '../flickr8k/Images'
    process.prepare()
    process.train()


if __name__ == '__main__':
    example()
