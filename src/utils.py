import torch
import torch.optim as optim

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OPTIMIZER_FACTORY = {
    'adam': optim.Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'gd': optim.SGD,
    'sparseadam': optim.SparseAdam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop
}