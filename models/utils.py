import torch.nn as nn

activations = {
    'none': None,
    'relu6': nn.ReLU6()
}


def str2act(s):
    if s in activations:
        return activations[s]
    else:
        raise ValueError('Unknown actiovation: ' + s)
