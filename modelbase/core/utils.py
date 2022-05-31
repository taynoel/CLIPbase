import yaml
from yacs.config import CfgNode

def read_param_yaml(param_yaml: str) -> CfgNode:
    with open(param_yaml, encoding='utf8') as fil:
        cfg = yaml.load(fil.read(), Loader=yaml.SafeLoader)
    return CfgNode(cfg)