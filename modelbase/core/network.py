from abc import ABCMeta, abstractmethod
from torch import Tensor, nn, device, load


class AbstractNetwork(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(AbstractNetwork, self).__init__()
        self.hook_result = {}

    def load_pretrained_param_from_file(self, file_name: str):
        device_info = device("cuda" if next(self.parameters()).is_cuda else "cpu")
        try:
            pretrainedDict: dict = load(file_name, map_location=device_info.type)
            modelDict = self.state_dict()
            pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
            modelDict.update(pretrainedDict)
            self.load_state_dict(modelDict)
        except FileNotFoundError:
            print("Can't load pre-trained parameter files")

    def get_layer_value(self, name):
        """ Network hook
        Can be used as such:
        network.register_forward_hook(self.get_layer_value('name'))
        """
        def hook(model, input: Tensor, output: Tensor):
            self.hook_result[name] = output.detach()
        return hook

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
